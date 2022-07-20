class Packet:
    ...


class Request(Packet):
    ...


class FrameRequest(Request):
    def __init__(self, framesize):
        self.framesize = framesize


class Frame(Packet):
    def __init__(self, frame):
        self.frame = frame


class LastFrame(Frame):
    def __init__(self, frame, valid_samples):
        self.frame = frame
        self.valid_samples = valid_samples


class Pipeline:
    def __init__(self, *nodes):
        self.nodes = nodes

    def __repr__(self):
        return str([node.__class__.__name__ for node in self])

    def __iter__(self):
        return iter(self.nodes)

    @property
    def samplerate_decider(self):
        for node in self:
            if isinstance(node, SamplerateDecider):
                return node

    def run(self):
        self.samplerate_decider.run()

    def setup(self):
        for node in self:
            node.setup()

    def reset(self):
        for node in self:
            node.reset()

    def __or__(self, other):
        if isinstance(other, Pipeline):
            self.nodes[-1]._downstream = other.nodes[0]
            other.nodes[0]._upstream = self.nodes[-1]
            return Pipeline(*self.nodes, *other.nodes)
        if isinstance(other, Node):
            self.nodes[-1]._downstream = other
            other._upstream = self.nodes[-1]
            return Pipeline(*self.nodes, other)
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, Pipeline):
            self.nodes[0]._upstream = other.nodes[-1]
            other.nodes[-1]._downstream = self.nodes[0]
            return Pipeline(*other.nodes, *self.nodes)
        if isinstance(other, Node):
            self.nodes[0]._upstream = other
            other._downstream = self.nodes[0]
            return Pipeline(other, *self.nodes)
        return NotImplemented


class Node:
    """Generic pipeline Node."""

    def __init__(self):
        self._is_ready = False
        self.__upstream = None
        self.__downstream = None

    def __or__(self, other):
        if isinstance(other, Node):
            self._downstream = other
            other._upstream = self
            return Pipeline(self, other)
        return NotImplemented

    def setup(self):
        """Run setup for this Node."""
        self._is_ready = True

    def reset(self):
        """Reset this Node."""
        self._is_ready = False

    def process(self, frame):
        """Process one frame of data."""
        return frame

    @property
    def _upstream(self):
        """Wrap for the upstream node object."""
        return self.__upstream

    @_upstream.setter
    def _upstream(self, node):
        """Set the upstream node, running update code."""
        self.__upstream = node
        self._upstream_changed()

    def _upstream_changed(self):
        """Code to run when the upstream is changed."""
        pass

    @property
    def _downstream(self):
        """Wrap for the downstream node object."""
        return self.__downstream

    @_downstream.setter
    def _downstream(self, node):
        """Set the downstream node, running update code."""
        self.__downstream = node
        self._downstream_changed()

    def _downstream_changed(self):
        """Code to run when the downstream is changed."""
        pass

    def push(self, packet):
        """Give a packet of data as the input to this node.

        Processes a frame and passes it along down the pipeline.
        """
        if isinstance(packet, Frame):
            packet = self.process(packet)

        if packet is None:
            return

        if self._downstream is not None:
            # We should push the frame down the pipeline, and return the final result.
            return self._downstream.push(packet)
        else:
            # This is the end of the pipeline, so return the frame.
            return packet

    def request(self, packet):
        """Request one frame of output from this object.

        Requests a frame from the upstream or this node, process it,
        then return the frame.
        """
        if self._upstream is not None:
            # There's an upstream node, ask it to handle the packet first.
            packet = self._upstream.request(packet)

        if isinstance(packet, Request):
            if isinstance(packet, FrameRequest):
                # Either there's no upstream node, or it could not handle the request.
                # Process this request here, then return the generated frame.
                return self.process(packet)


class SamplerateDecider(Node):
    """Base class for nodes that have an externally defined samplerate.

    Instances of this type will decide the samplerate for an entire pipeline,
    by telling the SamplerateFollowers what samplerate they are operating at.
    """
    def __init__(self, samplerate, **kwargs):
        super().__init__(**kwargs)
        self.samplerate = samplerate

    def _upstream_changed(self):
        super()._upstream_changed()
        if isinstance(self._upstream, SamplerateFollower):
            self._upstream._set_samplerate_direction('downstream')

    def _downstream_changed(self):
        super()._downstream_changed()
        if isinstance(self._downstream, SamplerateFollower):
            self._downstream._set_samplerate_direction('upstream')


class SamplerateFollower(Node):
    """Base class for nodes which have no external samplerate.

    Instances of this class will look for a samplerate elsewhere in the pipeline.
    This can be found either upstream when the instance is receiving data from
    a SamperateDecider, or downstream when the instance will send data to a SamplerateDecider.
    This is managed automatically when assembling the pipeline, no user interaction required.
    """
    def __init__(self, **kwargs):
        self._samplerate_direction = None
        super().__init__(**kwargs)

    def _upstream_changed(self):
        """Clear any stored samplerate direction since it might no longer be valid."""
        super()._upstream_changed()
        self._samplerate_direction = None

    def _downstream_changed(self):
        """Clear any stored samplerate direction since it might no longer be valid."""
        super()._downstream_changed()
        self._samplerate_direction = None

    @property
    def samplerate(self):
        direction = self._get_samplerate_direction()
        if direction == 'upstream':
            samplerate = self._upstream.samplerate
            try:
                _, samplerate = samplerate
            except TypeError:
                pass
            return samplerate
        if direction == 'downstream':
            samplerate = self._downstream.samplerate
            try:
                samplerate, _ = samplerate
            except TypeError:
                pass
            return samplerate

    def _get_samplerate_direction(self, direction='both'):
        """Check if the samplerate can be found along the pipeline.

        This will run along the pipeline and see if there's a SamplerateDecider
        somewhere. If a direction is specified, it will only look in the given direction.
        If found, all other SamplerateFollowers will be updated.

        This is a complicated getter for the samplerate direction, parameterized with a
        search direction to allow recursive searches. Additionally, when the value is found
        it will be updated for all the relevant SamplerateFollowers.
        """
        if self._samplerate_direction is not None:
            return self._samplerate_direction

        if direction == 'both':
            self._samplerate_direction = self._get_samplerate_direction('upstream') or self._get_samplerate_direction('downstream')
            return self._samplerate_direction

        if direction == 'upstream':
            if isinstance(self._upstream, SamplerateDecider):
                self._set_samplerate_direction('upstream')
                return 'upstream'
            elif isinstance(self._upstream, SamplerateFollower):
                return self._upstream._get_samplerate_direction('upstream')
        elif direction == 'downstream':
            if isinstance(self._downstream, SamplerateDecider):
                self._set_samplerate_direction('downstream')
                return 'downstream'
            elif isinstance(self._downstream, SamplerateFollower):
                return self._downstream._get_samplerate_direction('downstream')
        else:
            raise ValueError(f"Search direction should be one of 'upstream' or 'downstream', got {direction}")
        return False

    def _set_samplerate_direction(self, direction):
        """Update the pipeline samplerate direction.

        This will go through the pipeline in one direction and update
        the samplerate search direction.
        It will always go through the pipeline in the opposite direction to the
        search direction. I.e., if diretcion='upstream', this will go downstream
        set all the samplerate directions to 'upstream', and vice versa.
        """
        if direction == 'upstream':
            self._samplerate_direction = 'upstream'
            if isinstance(self._downstream, SamplerateFollower):
                if self._downstream._samplerate_direction is None:
                    self._downstream._set_samplerate_direction('upstream')
                elif self._downstream._samplerate_direction == 'downstream':
                    raise ValueError('Conflicting pipeline samplerates!')
        elif direction == 'downstream':
            self._samplerate_direction = 'downstream'
            if isinstance(self._upstream, SamplerateFollower):
                if self._upstream._samplerate_direction is None:
                    self._upstream._set_samplerate_direction('downstream')
                elif self._upstream._samplerate_direction == 'upstream':
                    raise ValueError('Conflicting pipeline samplerates!')

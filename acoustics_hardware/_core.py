class Node:
    """Generic pipeline Node."""

    def __init__(self, input_node=None, output_node=None):
        self.__input = None
        self.__output = None
        if input_node is not None:
            self.insert_input(input_node)
        if output_node is not None:
            self.insert_output(output_node)

    def setup(self, pipeline=False):
        """Run setup for this Node.

        If the `pipeline` input is True, the entire pipeline will be setup.
        """
        if not pipeline:
            return
        if isinstance(self._input, Node):
            self._input.setup(True)
        if isinstance(self._output, Node):
            self._output.setup(True)

    def reset(self, pipeline=False):
        """Reset this Node.

        If the `pipeline` input is True, the entire pipeline will be reset.
        """
        if not pipeline:
            return
        if isinstance(self._input, Node):
            self._input.reset(True)
        if isinstance(self._output, Node):
            self._output.reset(True)

    def process(self, frame):
        """Process one frame of data."""
        return frame

    @property
    def _input(self):
        """Wrap for the input node object."""
        return self.__input

    @_input.setter
    def _input(self, input):
        """Set the input node, running update code."""
        self.__input = input
        self._input_changed()

    def _input_changed(self):
        """Code to run when the input is changed."""
        pass

    @property
    def _output(self):
        """Wrap for the output node object."""
        return self.__output

    @_output.setter
    def _output(self, output):
        """Set the output node, running update code."""
        self.__output = output
        self._output_changed()

    def _output_changed(self):
        """Code to run when the output is changed."""
        pass

    def insert_input(self, insert):
        """Insert a node between this node and it's input."""
        if isinstance(self._input, Node):
            self._input._output = insert
        insert._input = self._input
        insert._output = self
        self._input = insert

    def insert_output(self, insert):
        """Insert a node between this node and it's output."""
        if isinstance(self._output, Node):
            self._output._input = insert
        insert._output = self._output
        insert._input = self
        self._output = insert

    def input(self, frame):
        """Give a frame of data as the input to this node.

        Processes a frame and passes it along down the pipeline.
        """
        frame = self.process(frame)

        if self._output is not None:
            # We should push the frame down the pipeline, and return the final result.
            return self._output.input(frame)
        else:
            # This is the end of the pipeline, so return the frame.
            return frame

    def output(self, framesize):
        """Request one frame of output from this object.

        Requests a frame from the input or this node, processes it,
        then return the frame.
        """
        if self._input is not None:
            # There's an input node, ask it for a frame of the correct size, process it, and return down the pipeline
            frame = self._input.output(framesize)
            return self.process(frame=frame)
        else:
            # There's no input node, so self has to be a generator. Give the framesize to the process function, then return down the pipeline.
            return self.process(framesize=framesize)


class SamplerateDecider(Node):
    """Base class for nodes that have an externally defined samplerate.

    Instances of this type will decide the samplerate for an entire pipeline,
    by telling the SamplerateFollowers what samplerate they are operating at.
    """
    def __init__(self, samplerate, **kwargs):
        super().__init__(**kwargs)
        self.samplerate = samplerate

    def _input_changed(self):
        super()._input_changed()
        if isinstance(self._input, SamplerateFollower):
            self._input._set_samplerate_direction('downstream')

    def _output_changed(self):
        super()._output_changed()
        if isinstance(self._output, SamplerateFollower):
            self._output._set_samplerate_direction('upstream')


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

    def _input_changed(self):
        """Clear any stored samplerate direction since it might no longer be valid."""
        super()._input_changed()
        self._samplerate_direction = None

    def _output_changed(self):
        """Clear any stored samplerate direction since it might no longer be valid."""
        super()._output_changed()
        self._samplerate_direction = None

    @property
    def samplerate(self):
        direction = self._get_samplerate_direction()
        if direction == 'upstream':
            return self._input.samplerate
        if direction == 'downstream':
            return self._output.samplerate

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
            self._get_samplerate_direction('upstream') or self._get_samplerate_direction('downstream')
            return self._samplerate_direction

        if direction == 'upstream':
            if isinstance(self._input, SamplerateDecider):
                self._set_samplerate_direction('upstream')
                return True
            elif isinstance(self._input, SamplerateFollower):
                return self._input._get_samplerate_direction('upstream')
        elif direction == 'downstream':
            if isinstance(self._output, SamplerateDecider):
                self._set_samplerate_direction('downstream')
                return True
            elif isinstance(self._output, SamplerateFollower):
                return self._output._get_samplerate_direction('downstream')
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
            if isinstance(self._output, SamplerateFollower):
                if self._output._samplerate_direction is None:
                    self._output._set_samplerate_direction('upstream')
                elif self._output._samplerate_direction == 'downstream':
                    raise ValueError('Conflicting pipeline samplerates!')
        elif direction == 'downstream':
            self._samplerate_direction = 'downstream'
            if isinstance(self._input, SamplerateFollower):
                if self._input._samplerate_direction is None:
                    self._input._set_samplerate_direction('downstream')
                elif self._input._samplerate_direction == 'upstream':
                    raise ValueError('Conflicting pipeline samplerates!')

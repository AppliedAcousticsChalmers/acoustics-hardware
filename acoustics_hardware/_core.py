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



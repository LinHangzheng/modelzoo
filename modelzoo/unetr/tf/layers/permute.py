from tensorflow.keras.layers import Permute
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer
class PermuteLayer(BaseLayer):
    """Wrapper around the Keras layer that reshapes the input.
    
    Parameters
    ----------
    dims
    """

    def __init__(
        self, dims, boundary_casting=False, tf_summary=False, **kwargs
    ):
        super(PermuteLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.layer = Permute(
            dims, dtype=self.dtype_policy, name=self.name
        )

    def call(self, input, **kwargs):
        """Apply the reshape layer to an input.

        Args:
            inputs (Tensor): A tensor.

        Returns:
            Tensor: The tensor after reshape.
        """

        if self.boundary_casting:
            input = boundary_cast(input)
        output = self.layer(input)
        if self.tf_summary:
            output = summary_layer(output)
        return output

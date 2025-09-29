import jax
import jax.numpy as jnp
from flax import linen as nn

def flip_random_bit_real(x, key):
    """
    Flip a random bit in the real part of a complex64 scalar x.
    Returns the modified scalar and a new PRNG key.
    """
    real = jnp.real(x)
    imag = jnp.imag(x)
    
    real_int = real.view(jnp.int32)
    
    key, subkey = jax.random.split(key)
    bit_to_flip = jax.random.randint(subkey, (), 0, 32)
    
    mask = 1 << bit_to_flip
    real_flipped_int = real_int ^ mask
    real_flipped = real_flipped_int.view(jnp.float32)
    
    return jnp.array(real_flipped + 1j * imag, dtype=jnp.complex64), key

def flip_random_element_bit(arr, key):
    """
    Flip a random bit in the real part of a random element of a complex64 array.
    
    Args:
        arr: JAX array of complex64, any shape
        key: JAX PRNGKey
        
    Returns:
        new array with one random element modified
        updated key
    """
    # Flatten the array to pick a single element
    flat_arr = arr.ravel()
    
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (), 0, flat_arr.size)
    
    # Flip bit for the selected element
    new_val, key = flip_random_bit_real(flat_arr[idx], key)
    
    # Use jax.ops.index_update to replace the value
    flat_arr = flat_arr.at[idx].set(new_val)
    
    # Reshape back to original shape
    return flat_arr.reshape(arr.shape), key

def flip_bits_array(arr, key, error_rate=0.5):
    """
    Flip a random mantissa bit in the real part of each element with probability `error_rate`.
    
    Args:
        arr: complex64 JAX array of any shape
        key: JAX PRNGKey
        error_rate: probability each element is flipped
    
    Returns:
        new array, updated key
    """
    shape = arr.shape
    flat_arr = arr.ravel()
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Mask: which elements to flip
    flip_mask = jax.random.uniform(subkey1, flat_arr.shape) < error_rate
    
    # Random bits for each element
    bit_choices = jax.random.randint(subkey2, flat_arr.shape, 0, 23)
    
    # Vectorized bit-flip function
    def flip_if_masked(val, bit, mask):
        val_int = jnp.real(val).view(jnp.int32)
        mask_val = 1 << bit
        val_int_flipped = jnp.where(mask, val_int ^ mask_val, val_int)
        real_flipped = val_int_flipped.view(jnp.float32)
        return jnp.array(real_flipped + 1j * jnp.imag(val), dtype=jnp.complex64)
    
    # Vectorize over all elements
    flat_arr_flipped = jax.vmap(flip_if_masked)(flat_arr, bit_choices, flip_mask)
    
    return flat_arr_flipped.reshape(shape), key

class MyModule(nn.Module):
    def __call__(self, x, inject_noise=False):
        if inject_noise:
            key = self.make_rng("fault")
            x = x + jax.random.normal(key, x.shape)
        return x

# Example usage
key = jax.random.PRNGKey(0)
arr = jnp.array([[1.5 + 2j, 3.0 + 4j],
                 [5.0 + 6j, 7.0 + 8j]], dtype=jnp.complex64)

for _ in range(5):
    arr_flipped, _ = flip_bits_array(arr, key)
    print(arr_flipped)

variables = MyModule().init(key, jnp.ones((2,)))
out = MyModule().apply(variables, jnp.ones((2,)), inject_noise=True, rngs={"fault": key})
print(out)
out = MyModule().apply(variables, jnp.ones((2,)), inject_noise=False, rngs={"fault": key})
print(out)
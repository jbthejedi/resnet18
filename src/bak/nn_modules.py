import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        kernel_size, stride, padding, bias=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)

    def forward(self, x):
        """
        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1

        out.shape = (b, c_out, h_out, w_out)
        """
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        x_unfold = F.unfold( # -> (b, c_in*k*k, L), where L = h_out * w_out
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        x_unfold_t = x_unfold.transpose(1, 2) # (b, L, c_in*k*k)
        weights = self.weights.view(-1, c_in*k*k) # (c_out, c_in*k*k)
        out = x_unfold_t @ weights.t() # (b, L, c_out)
        out = out.transpose(1, 2)
        h_out = (h_in + 2 * p - k) // s + 1
        w_out = (w_in + 2 * p - k) // s + 1
        out = out.view(b, -1, h_out, w_out)
        return out

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding

        # x_unfold.shape => (b, c_in*k*k, L), where L = h_out * w_out
        x_unfold = F.unfold(x, kernel_size=k, stride=s, padding=p)
        x_unfold = x_unfold.view(b, c_in, k*k, -1)
        x_max, _ = x_unfold.max(dim=2)
        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1
        out = x_max.view(b, c_in, h_out, w_out)

        return out

class AvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        b, c_in, h_in, w_in = x.shape
        h_out, w_out = self.output_size
        kernel_h = h_in // h_out
        kernel_w = w_in // w_out
        kernel = (kernel_h, kernel_w)
        x_unfold = F.unfold( # (b, c_in*k*k, L), L = 1*1 = 1
            x,
            kernel_size=kernel,
            stride=kernel,
        )
        x_unfold = x_unfold.view(b, c_in, kernel_h*kernel_w, h_out, w_out)
        out = x_unfold.mean(dim=2)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_var = x.var(-1, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        x_hat = x_hat * self.gamma + self.beta

        return x_hat

class BatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features, # num_channels/num_feature_maps
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Affine = use learnable scale (gamma) and shift (beta)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        # If the model is training, than we calculate 
        # the current mean and var using the past mean
        # and var with momentum.
        if self.training:
            # x.shape = (b, c_in, h_in, w_in)
            # Calculate fe feature map the mean/var across all images in batch.
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)

            if self.track_running_stats:
                # Momentum => running_mean is more important than batch_mean
                # if mom = 0.1 => 1 - 0.1 = .9 and .9*running_mean + .1*batch_mean 
                # puts more importance on running_mean
                self.running_mean = (
                    (1 - self.momentum)*self.running_mean + self.momentum*batch_mean
                )
                self.running_var = (
                    (1 - self.momentum)*self.running_var + self.momentum*batch_var
                )
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # x_hat = x - mu / sqrt(sigma)
        x_norm = (
            (x - mean[None, :, None, None])
            / torch.sqrt(var[None, :, None, None] + self.eps)
        )
        if self.affine:
            x_norm = (
                (self.gamma[None, :, None, None] * x_norm)
                + self.beta[None, :, None, None]
            )

        return x


            

def test_modules():
    # -----------
    # Conv2d
    # -----------
    module_name = "Conv2d"
    print(f"Testing {module_name}")
    b, c, w, h = 4, 3, 16, 16
    in_tensor = torch.ones(b, c, w, h)
    k, s, p = 3, 1, 1
    module = Conv2d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")
    assert out_tensor.shape == (4, 6, 16, 16), f"Failed. output.shape {out_tensor.shape}"

    # -----------
    # MaxPool2d
    # -----------
    module_name = "MaxPool2d"
    print(f"Testing {module_name}")
    b, c, w, h = 4, 3, 16, 16
    in_tensor = torch.ones(b, c, w, h)
    k, s, p = 3, 1, 1
    module = MaxPool2d(
        kernel_size=3,
        stride=1,
        padding=1,
    )
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")
    assert out_tensor.shape == (b, c, w, h), f"Failed. output.shape {out_tensor.shape}"

    # -----------
    # AvgPool2d
    # -----------
    module_name = "AvgPool2d"
    print(f"Testing {module_name}")
    b, c, w, h = 4, 20, 16, 16
    in_tensor = torch.ones(b, c, w, h)
    k, s, p = 3, 1, 1
    output_size=(1, 1)
    module = AvgPool2d(output_size=output_size)
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")

    assert out_tensor.shape == (b, c, output_size[0], output_size[1]), f"Failed. output.shape {out_tensor.shape}"

    # -----------
    # LayerNorm
    # -----------
    module_name = "LayerNorm"
    print(f"Testing {module_name}")
    # b=batch, t=token, d=dim_token_embedding
    b, t, d = 4, 8, 32
    in_tensor = torch.ones(b, t, d)
    module = LayerNorm(dim=d)
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")

    assert out_tensor.shape == (b, t, d), f"Failed. output.shape {out_tensor.shape}"

    # -----------
    # BatchNorm2d
    # -----------
    module_name = "BatchNorm2d"
    print(f"Testing {module_name}")
    # b=batch, c=num_channels_in
    b, c, h_in, w_in = 4, 8, 32, 32
    in_tensor = torch.ones(b, c, h_in, w_in)
    module = BatchNorm2d(
        num_features=c,
        # track_running_stats,
    )
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")

    assert out_tensor.shape == (b, c, h_in, w_in), f"Failed. output.shape {out_tensor.shape}"

def main():
    test_modules()

if __name__ == '__main__':
    main()


import torch

def pytorch_check():
    print("PyTorch version:", torch.__version__)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

def tensor_basic():
    # simple tensor 

    # 0 - d : scalar tensor. 
    x=torch.tensor(5.0)
    print("Tensor x:", x) # output: tensor(5.)
    print("Tensor x shape:", x.shape) # output: torch.Size([])
    print("Tensor x dimensions:", x.ndim) # output: 0
    print("Tensor x data item:", x.item()) # output: 5.0


    # 1-d : vector tensor, 
    y=torch.tensor([1.0,2.0,3.0])
    print("Tensor y:", y) # output: tensor([1., 2., 3.])
    print("Tensor y shape:", y.shape) # output: torch.Size([3])
    print("Tensor y dimensions:", y.ndim) # output: 1
    for i in range(y.shape[0]):
        print(f"Tensor y element {i}:", y[i].item()) # output: 1.0, 2.0, 3.0
    
    # 2-d : matrix tensor
    z=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    print("Tensor z:", z) # output: tensor([[1., 2., 3.],
    #        [4., 5., 6.]])
    print("Tensor z shape:", z.shape) # output: torch.Size([2, 3])
    print("Tensor z dimensions:", z.ndim) # output: 2
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            print(f"Tensor z element ({i},{j}):", z[i,j].item()) # output: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0

    # 3-d : 3D tensor
    w=torch.tensor([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])
    print("Tensor w:", w) # output: tensor([[[1., 2.],
    #         [3., 4.]],    
    #        [[5., 6.],
    #         [7., 8.]]])
    print("Tensor w shape:", w.shape) # output: torch.Size([2, 2, 2])
    print("Tensor w dimensions:", w.ndim) # output: 3
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            for k in range(w.shape[2]):
                print(f"Tensor w element ({i},{j},{k}):", w[i,j,k].item()) # output: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0

    # flattening a tensor
    for i in w.flatten():
        print("Flattened tensor w element:", i.item()) # output: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0


    # randome tensor
    random_tensor=torch.rand(3,4)
    print("Random tensor:", random_tensor)
    print("Random tensor type", random_tensor.dtype)

    # comman image size of the random tensor
    random_image_tensor=torch.rand(size=(224,224,3))
    print("Random image tensor shape:", random_image_tensor.shape)
    print("Random image tensor ndim:", random_image_tensor.ndim)


# convert to the HWC format to the CHW format
    chw_tensor=random_image_tensor.permute(2,0,1)
    print("CHW tensor shape:", chw_tensor.shape)
    print("CHW tensor ndim:", chw_tensor.ndim)


# zero tensor
    zero_tensor=torch.zeros(size=(3,4))
    print("Zero tensor:", zero_tensor)
    print("zero tensor type:", zero_tensor.dtype)

# all one tensor
    ones_tensor=torch.ones(size=(3,4))
    print("Ones tensor:", ones_tensor)
    print("Ones tensor type:", ones_tensor.dtype)


# torch range 
    zero_to_ten_tensor=torch.range(start=0,end=10,step=1)
    print("Zero to ten tensor:", zero_to_ten_tensor)

# tensor datatype
    float_32_tensor=torch.tensor([1.0,2.0,3.0],dtype=torch.float32,device=None, requires_grad=False )
    print("Float 32 tensor:", float_32_tensor)
    print("Float 32 tensor type:", float_32_tensor.dtype)

    float_16_tensor=torch.tensor([3.0,6.0,9.0],dtype=torch.float16)
    print("Float 16 tensor:", float_16_tensor)
    print("Float 16 tensor type:", float_16_tensor.dtype)

# tensors about device
    mps_tensor=torch.tensor([1.0,2.0,3.0],dtype=torch.float32,device="mps")
    print("MPS tensor:", mps_tensor)
    print("MPS tensor type:", mps_tensor.dtype)
    print("MPS tensor device:", mps_tensor.device)

    cpu_tensor=torch.tensor([1.0,2.0,3.0],dtype=torch.float32,device="cpu")
    print("CPU tensor:", cpu_tensor)
    print("CPU tensor type:", cpu_tensor.dtype)
    print("CPU tensor device:", cpu_tensor.device)

# common issue is abotu dtype and device , and shape





if __name__ == "__main__":
    pytorch_check()
    tensor_basic()
import torch

# Свёрточная сеть для 8K и ln=100 и ядром 17 - с батч-нормом в начале и перед активациями
class New_Morse_Signal_Level_k17_8k_Deep_Rolling_BNorm(torch.nn.Module):
    def __init__(self, n_input_channels, n_class, act):
        super(New_Morse_Signal_Level_k17_8k_Deep_Rolling_BNorm, self).__init__()
        self.n_input_channels = n_input_channels
        self.n_class = n_class
        # начальная нормализация
        self.bnorm_ = torch.nn.BatchNorm1d(n_input_channels)
        # 0 group, in 2 х 100, out 64 x 84
        n_neurons_0 = 64
        self.conv_0 = torch.nn.Conv1d(in_channels=n_input_channels,
                                    out_channels=n_neurons_0,
                                    kernel_size=17,
                                    padding=0,
                                    stride=1)
        
        self.bnorm_0 = torch.nn.BatchNorm1d(n_neurons_0)
        self.ac_0 = act # torch.nn.Tanh() # сделал тут другую функцию активации с боле плавной характеристикой
        
        # 1st group, in 64 x 84, out 128 x 42
        n_neurons_1 = 128
        self.conv_1_1 = torch.nn.Conv1d(in_channels=n_neurons_0,
                                    out_channels=n_neurons_1,
                                    kernel_size=9,
                                    padding=4,
                                    stride=1)

        self.conv_1_2 = torch.nn.Conv1d(in_channels=n_neurons_1,
                                    out_channels=n_neurons_1,
                                    kernel_size=9,
                                    padding=4,
                                    stride=1)
        
        self.mp_1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.bnorm_1 = torch.nn.BatchNorm1d(n_neurons_1) 
        self.ac_1 = act # разные функции на выбор
         
        # 2nd group, in 128 x 42, out 256 x 21
        n_neurons_2 = 256        
        self.conv_2_1 = torch.nn.Conv1d(in_channels=n_neurons_1,
                                    out_channels=n_neurons_2,
                                    kernel_size=5,
                                    padding=2,   
                                    stride=1)

        self.conv_2_2 = torch.nn.Conv1d(in_channels=n_neurons_2,
                                    out_channels=n_neurons_2,
                                    kernel_size=5,
                                    padding=2,   
                                    stride=1)
        
        self.mp_2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.bnorm_2 = torch.nn.BatchNorm1d(n_neurons_2)         
        self.ac_2 = act # было LeakyReLU
        
        self.dr_2 = torch.nn.Dropout(p=0.5)

        # 3rd group, in 256 x 21, out 512 x 7
        n_neurons_3 = 512        
        self.conv_3_1 = torch.nn.Conv1d(in_channels=n_neurons_2,
                                    out_channels=n_neurons_3,
                                    kernel_size=3,
                                    padding=1,  
                                    stride=1)
        
        self.mp_3 = torch.nn.MaxPool1d(kernel_size=3, stride=3)
        self.bnorm_3 = torch.nn.BatchNorm1d(n_neurons_3)        
        self.ac_3 = act    
        
        # 4th group, in 512 x 7, out 1024 x 1
        n_neurons_4 = 1024
        self.conv_4_1 = torch.nn.Conv1d(in_channels=n_neurons_3,
                                    out_channels=n_neurons_4,
                                    kernel_size=3,
                                    padding=1,  
                                    stride=1)
        
        self.mp_4 = torch.nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv_4_2 = torch.nn.Conv1d(in_channels=n_neurons_4,
                                    out_channels=n_neurons_4,
                                    kernel_size=3,
                                    padding=0,  
                                    stride=1)        
        
        self.bnorm_4 = torch.nn.BatchNorm1d(n_neurons_4)        
        self.ac_4 = act
        
        # 5th group, in 1024 x 1, out 1024 x 1
        self.dr_5 = torch.nn.Dropout(p=0.7) # on-off

        # 6h group, может надо перед ним еще один fc добавить? in 256 x 8 out 2048 -> 18
        self.fc_6 = torch.nn.Linear(n_neurons_4, n_neurons_4//4)
        self.dr_6 = torch.nn.Dropout(p=0.5)
        self.fc_7 = torch.nn.Linear(n_neurons_4//4, n_class)
       
        # output 
        self.sm = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.bnorm_(x) # начальная нормализация
        
        x = self.conv_0(x)
        x = self.bnorm_0(x)
        x = self.ac_0(x)
        
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.mp_1(x)
        x = self.bnorm_1(x)        
        x = self.ac_1(x)

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.mp_2(x)
        x = self.bnorm_2(x)        
        x = self.ac_2(x)
        
        x = self.dr_2(x)
        
        x = self.conv_3_1(x)
        x = self.mp_3(x)
        x = self.bnorm_3(x)        
        x = self.ac_3(x)
        
        x = self.conv_4_1(x)
        x = self.mp_4(x)
        x = self.conv_4_2(x)
        x = self.bnorm_4(x)        
        x = self.ac_4(x)
        
        x = self.dr_5(x)

        x = x.view(x.size(0), x.size(1) * x.size(2))
        x = self.fc_6(x)
        x = self.dr_6(x)
        x = self.fc_7(x)
        
        x = x.unsqueeze(2) # нужна после Dense-слоя

        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

# ResNet
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', 
                 use_batch_norm=True, use_drop_out=False, d_out_p=0.5):
        super(BasicBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_drop_out = use_drop_out
        self.d_out_p = d_out_p
        self.act  = torch.nn.ReLU() #torch.nn.Mish() #torch.nn.ReLU()
        
        self.conv1 = torch.nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(planes)
        self.d_out1 = torch.nn.Dropout1d(d_out_p)
        self.conv2 = torch.nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(planes)
        self.d_out2 = torch.nn.Dropout1d(d_out_p)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x: torch.nn.functional.pad(x[:, :, ::2], \
                                            (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = torch.nn.Sequential(
                     torch.nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     torch.nn.BatchNorm1d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        if self.use_drop_out:
            out = self.d_out1(out) 
        out = self.act(out)
        out = self.conv2(out)
        
        if self.use_batch_norm:
            out = self.bn2(out)
        if self.use_drop_out:
            out = self.d_out2(out)
          
        out += self.shortcut(x)
        out = self.act(out)
        return out

class ResNet1d(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=4, 
                 use_batch_norm=True, use_drop_out=False, d_out_p=0.5, num_channels=8):
        super(ResNet1d, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_drop_out = use_drop_out
        self.d_out_p = d_out_p
        self.in_planes = 16
        self.act  = torch.nn.ReLU() #torch.nn.Mish() #torch.nn.ReLU()

        self.conv1 = torch.nn.Conv1d(num_channels, 16, kernel_size=3, stride=1, padding=1, bias=False) # 2 channels
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.d_out1 = torch.nn.Dropout1d(d_out_p)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = torch.nn.Linear(64, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                use_batch_norm=self.use_batch_norm,
                               use_drop_out=self.use_drop_out, 
                               d_out_p=self.d_out_p))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        if self.use_drop_out:
            out = self.d_out1(out)
          
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool1d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.unsqueeze(2) # кажется она нужна
        return out
                   
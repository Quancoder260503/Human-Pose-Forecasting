import torch
import torch.nn as nn
import torch.nn.functional as F
from click.termui import hidden_prompt_func
from jupyter_client import launch_kernel
from nbformat.v3 import new_heading_cell


class ConvNetA(nn.Module):
    def __init__(self, num_out_classes):
        super(ConvNetA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 128,  kernel_size = 9, padding = 4)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 9, padding = 4)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 9, padding = 4)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 5, padding = 2)
        self.conv5 = nn.Conv2d(in_channels = 32,  out_channels = 512, kernel_size = 9, padding = 4)
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 1)
        self.conv7 = nn.Conv2d(in_channels = 512, out_channels = num_out_classes, kernel_size = 1)

    def forward(self, image):
        '''
        :param image: tensor of shape (B, 3, 368, 368)
        :return: (B, num_joints, 46, 46)
        '''
        image = self.pool1(F.relu(self.conv1(image))) # (3, 368, 368)   -> (128, 184, 184)
        image = self.pool2(F.relu(self.conv2(image))) # (128, 184, 184) -> (128,  92, 92)
        image = self.pool3(F.relu(self.conv3(image))) # (128, 92, 92)   -> (128,  46, 46)
        image = F.relu(self.conv4(image)) # (128, 46, 46)   -> (32, 46, 46)
        image = F.relu(self.conv5(image)) # (32, 46, 46)   -> (512, 46, 46)
        image = F.relu(self.conv6(image)) # (512, 46, 46)  -> (512, 46, 46)
        joints_heat_map = self.conv7(image) # (512, 46, 46) -> (num_joints, 46, 46)
        return joints_heat_map


class ConvNetB(nn.Module):
    def __init__(self):
        super(ConvNetB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 9, padding = 4)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 9, padding = 4)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 9, padding = 4)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 1, padding = 0)

    def forward(self, image):
        '''
        :param image: tensor of shape (B, 3, 368, 368)
        :return: (B, 32, 46, 46)
        '''
        image = self.pool1(F.relu(self.conv1(image)))
        image = self.pool2(F.relu(self.conv2(image)))
        image = self.pool3(F.relu(self.conv3(image)))
        image = F.relu(self.conv4(image))
        return image

class ConvNetC(nn.Module):
    def __init__(self, num_out_classes):
        super(ConvNetC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 48, out_channels = 128, kernel_size = 11, padding = 5)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 11, padding = 5)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 11, padding = 5)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, padding = 0)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = num_out_classes, kernel_size = 1, padding = 0)

    def forward(self, image):
        '''
        :param image: tensor of shape (B, 3, 368, 368)
        :return: (B, num_joints, 46, 46)
        '''
        image = F.relu(self.conv1(image))
        image = F.relu(self.conv2(image))
        image = F.relu(self.conv3(image))
        image = F.relu(self.conv4(image))
        image = self.conv5(image)
        return image

class ConvLSTMPM(nn.Module):
    def __init__(self, num_out_classes = 13 + 1, time_step = 7):
        super(ConvLSTMPM, self).__init__()
        self.num_out_classes = num_out_classes
        self.time_step       = time_step
        self.conv_net_A  = ConvNetA(num_out_classes)
        self.conv_net_B  = ConvNetB()
        self.conv_net_C  = ConvNetC(num_out_classes)
        self.pool_center = nn.AvgPool2d(kernel_size = 8, stride = 8)

        # LSTM, may adapt better architecture later
        self.x_conv_inp_gate = nn.Conv2d(in_channels = 32 + 1 + num_out_classes, out_channels = 48, kernel_size = 3, padding = 1)
        self.h_conv_inp_gate = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1)

        self.x_conv_for_gate = nn.Conv2d(in_channels = 32 + 1 + num_out_classes, out_channels = 48, kernel_size = 3, padding = 1)
        self.h_conv_for_gate = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1)

        self.x_conv_can_gate  = nn.Conv2d(in_channels = 32 + 1 + num_out_classes, out_channels = 48, kernel_size = 3, padding = 1)
        self.h_conv_can_gate  = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1)

        self.x_conv_out_gate = nn.Conv2d(in_channels = 32 + 1 + num_out_classes, out_channels = 48, kernel_size = 3, padding = 1)
        self.h_conv_out_gate = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 3, padding = 1)

        self.x_conv_inp_gate_init  = nn.Conv2d(in_channels = 32 + 1 + num_out_classes, out_channels = 48, kernel_size = 3, padding = 1)
        self.x_conv_out_gate_init  = nn.Conv2d(in_channels = 32 + 1 + num_out_classes, out_channels = 48, kernel_size = 3, padding = 1)
        self.x_conv_can_gate_init = nn.Conv2d(in_channels = 32 + 1 + num_out_classes, out_channels = 48, kernel_size = 3, padding = 1)

    def lstm(self, heatmap, features, center_map, hidden_t_prev, cell_t_prev):
        '''
        :param heatmap: tensor of shape (B, num_joints, 46, 46)
        :param features: tensor of shape (B, 32, 46, 46)
        :param center_map: tensor of shape (B, 1, 46, 46)
        :param hidden_t_prev: tensor of shape (B, 48, 46, 46), short term memory
        :param cell_t_prev: tensor of shape   (B, 48, 46, 46), long term memory
        :return:
          cell_t : tensor of shape (B, 48, 46, 46), the next cell state
          hidden_t : tensor of shape (B, 48, 46, 46), the next hidden state
        '''

        xt = torch.cat([heatmap, features, center_map], dim = 1) # (32 + num_joints + 1, 46, 46)

        # Candidate gate
        cx = self.x_conv_can_gate(xt)
        ch = self.h_conv_can_gate(hidden_t_prev)
        c_tilde = F.tanh(cx + ch)

        # Input gate
        ix = self.x_conv_inp_gate(xt)
        ih = self.h_conv_inp_gate(hidden_t_prev)
        i_gate = F.sigmoid(ix + ih)

        # Forget gate
        fx = self.x_conv_for_gate(xt)
        fh = self.h_conv_for_gate(hidden_t_prev)
        f_gate = F.sigmoid(fx + fh)

        # Output gate
        ox = self.x_conv_out_gate(xt)
        oh = self.h_conv_out_gate(hidden_t_prev)
        o_gate = F.sigmoid(ox + oh)

        cell_t = f_gate * cell_t_prev + i_gate * c_tilde

        hidden_t = F.tanh(cell_t) * o_gate

        return cell_t, hidden_t

    def lstm_stage_0(self, image):
        '''
        :param image: tensor of shape (B, 3, 368, 368)
        :return:
          cell_t : tensor of shape (B, 48, 46, 46)
          hidden_t : tensor of shape (B, 48, 46, 46)
        '''
        cx = self.x_conv_can_gate_init(image)
        ix = self.x_conv_inp_gate_init(image)
        ox = self.x_conv_inp_gate_init(image)

        cx = F.tanh(cx)
        ix = F.sigmoid(ix)
        ox = F.sigmoid(ox)

        cell_1 = F.tanh(cx * ix)
        hidden_1 = cell_1 * ox
        return cell_1, hidden_1

    def stage_A(self, image, cmap):
        """
        :param image: tensor of shape (B, 3, 368, 368)
        :param cmap: tensor of shape (B, 1, 368, 368)
        :return:
         heatmap, tensor of shape (B, num_joints, 46, 46)
         cell_t   : tensor of shape (B, 48, 46, 46)
         hidden_t : tensor of shape (B, 48, 46, 46)
        """
        initial_heatmap = self.conv_net_A(image) # (3, 368, 368) -> (num_joints, 46, 46)
        features        = self.conv_net_B(image) # (3, 368, 368) -> (32, 46, 46)
        center_map      = self.pool_center(cmap)

        x = torch.cat([initial_heatmap, features, center_map], dim = 1)
        cell_1, hidden_1 = self.lstm_stage_0(x)
        heatmap = self.conv_net_C(hidden_1)
        return initial_heatmap, heatmap, cell_1, hidden_1

    def stage_B(self, image, cmap, heatmap, cell_prev, hidden_prev):
        '''
        :param image: tensor of shape (B, 3, 368, 368)
        :param cmap: tensor of shape (B, 1, 368, 368)
        :param heatmap: tensor of shape (B, num_joints, 46, 46)
        :param cell_prev:  tensor of shape (B, 48, 46, 46)
        :param hidden_prev: tensor of shape (B, 48, 46, 46)
        :return:
          new_heatmap : a tensor of shape (B, num_joints, 46, 46)
          cell_t : tensor of shape (B, 48, 46, 46)
          hidden_t : tensor of shape (B, 48, 46, 46)
        '''

        features = self.conv_net_B(image)
        center_map = self.pool_center(cmap)

        cell_t, hidden_t = self.lstm(heatmap, features, center_map, hidden_prev, cell_prev)
        new_heatmap      = self.conv_net_C(hidden_t)
        return new_heatmap, cell_t, hidden_t

    def forward(self, images, center_map):
        '''
        Stack the channels with the time to reduce dimensionality
        :param images: a tensor of shape (B, T * 3, 368, 368)
        :param center_map: a tensor of shape (B, 1, 368, 368)
        :return:
          heat_maps : a tensor of shape (B, T, num_joints, 46, 46)
        '''
        heatmaps = []
        first_frame = images[:, 0 : 3, :, :]
        initial_heatmap, heatmap, cell, hidden = self.stage_A(first_frame, center_map)
        heatmaps.append(initial_heatmap)
        heatmaps.append(heatmap)

        for ind in range(1, self.time_step):
            image = images[:, (ind * 3) : (3 * ind + 3), :, :]
            heatmap, cell, hidden = self.stage_B(image, center_map, heatmap, cell, hidden)
            heatmaps.append(heatmap)
        return torch.stack(heatmaps, dim = 1)

# if __name__ == '__main__':
#     net = ConvLSTMPM(time_step = 4)
#     a = torch.randn(2, 12, 368, 368)
#     c = torch.randn(2, 1,  368, 368)
#
#     maps = net(a, c)
#     print(maps.shape)



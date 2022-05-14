import pandas as pd
import json
import h5py
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

writer = SummaryWriter()
torch.cuda.empty_cache()

data_path = "samples"


def plot_timeseq(eventspiketensor, frame_size_ms, segment):
    event_activation = eventspiketensor.sum(axis=(1, 2, 3))
    step = frame_size_ms * 1e-3  # Convert to second
    t_offset = step / 2
    t_frames = t_offset + step * np.arange(event_activation.shape[0])
    fig, ax = plt.subplots(figsize=(2, 0.5))
    ax.plot(t_frames, event_activation)
    if segment is not None and (segment[0] >= 0 and segment[1] >= 0):
        # print(segment)
        ax.axvline(segment[0], c='k')
        ax.axvline(segment[1], c='k')
    ax.set_xlim((t_frames[0], t_frames[-1]))
    max_y = event_activation.max()
    ax.set_ylim((0, max_y))
    return fig, ax


if not os.path.exists(data_path):
    os.makedirs(data_path)
    with open('sets.json', 'r') as myfile:
        data = myfile.read()
        obj = json.loads(data)
        # train values
        for i in obj['train']:
            with h5py.File("roi_mouth/" + i + ".h5", "r") as f:
                a_group_key = list(f.keys())[0]
                train_data = np.array(list(f[a_group_key]))
                train_values = obj['train'][i]
                plot_timeseq(train_data, 10, train_values)
                if (train_values[0] == -1):
                    plt.savefig(data_path + "/" + i + "&" + "no" + ".png")
                    plt.close()
                    plt.clf()
                else:
                    plt.savefig(
                        data_path + "/" + i + "&" + str(int(train_values[0])) + str(int(train_values[1])) + ".png")
                    plt.close()
                    plt.clf()
        # validation values
        for i in obj['validation']:
            with h5py.File("roi_mouth/" + i + ".h5", "r") as f:
                a_group_key = list(f.keys())[0]
                valid_data = np.array(list(f[a_group_key]))
                valid_values = obj['validation'][i]
                plot_timeseq(valid_data, 10, valid_values)
                if (valid_values[0] == -1):
                    plt.savefig(data_path + "/" + i + "&" + "no" + ".png")
                    plt.close()
                    plt.clf()
                else:
                    plt.savefig(
                        data_path + "/" + i + "&" + str(int(valid_values[0])) + str(int(valid_values[1])) + ".png")
                    plt.close()
                    plt.clf()
        # test values
        for i in obj['test']:
            with h5py.File("roi_mouth/" + i + ".h5", "r") as f:
                a_group_key = list(f.keys())[0]
                test_data = np.array(list(f[a_group_key]))
                test_values = obj['test'][i]
                plot_timeseq(test_data, 10, test_values)
                if (test_values[0] == -1):
                    plt.savefig(data_path + "/" + i + "&" + "no" + ".png")
                    plt.close()
                    plt.clf()
                else:
                    plt.savefig(
                        data_path + "/" + i + "&" + str(int(test_values[0])) + str(int(test_values[1])) + ".png")
                    plt.close()
                    plt.clf()

image_fns = os.listdir(data_path)

image_fns_train, image_fns_test = train_test_split(image_fns, random_state=0)
image_fns_valid, image_fns_valid = train_test_split(image_fns_train, random_state=0)

image_ns = [image_fn.split("&")[1].split(".p")[0] for image_fn in image_fns]
image_ns = "".join(image_ns)
letters = sorted(list(set(list(image_ns))))

vocabulary = ["-"] + letters
idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
char2idx = {v: k for k, v in idx2char.items()}

batch_size = 16


class Time(Dataset):

    def __init__(self, data_dir, image_fns):
        self.data_dir = data_dir
        self.image_fns = image_fns

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.data_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = self.transform(image)
        text = image_fn.split("&")[1].split(".p")[0]
        return image, text

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)


trainset = Time(data_path, image_fns_train)
testset = Time(data_path, image_fns_test)
validset = Time(data_path, image_fns_valid)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size,  shuffle=False)
valid_loader = DataLoader(validset, batch_size=batch_size,  shuffle=False)

results_test = pd.DataFrame(columns=['actual', 'prediction'])
results_train = pd.DataFrame(columns=['actual', 'prediction'])
results_valid = pd.DataFrame(columns=['actual', 'prediction'])

image_batch, text_batch = iter(train_loader).next()

num_chars = len(char2idx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class RNN(nn.Module):
    def __init__(self, vocab_size,lstm=True,batch_size=16):
        super(RNN, self).__init__()
        self.lstm=lstm
        self.batch_size=batch_size

        if lstm:
            self.hidden = self.init_hidden()
            self.lstm = nn.LSTM(input_size=150, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        else:
            self.gru = nn.GRU(input_size=150, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(256, vocab_size)

    def init_hidden(self):
        h = Variable(torch.zeros(2, self.batch_size, 256)).cuda()
        c = Variable(torch.zeros(2, self.batch_size, 256)).cuda()
        return h, c

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1)
        if self.lstm:
            output, _ = self.lstm(x,self.hidden)
        else:
            output, _ = self.gru(x)
        x = self.linear(output)
        x = x.permute(1, 0, 2)
        return x

class BiLSTM_Attention(nn.Module):
    def __init__(self,vocab_size):
        super(BiLSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(150, 256, bidirectional=True)
        self.out = nn.Linear(256 * 2, vocab_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, 256 * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.cpu().data.numpy()

    def forward(self, X):
        x =X.permute(0, 3, 1, 2)
        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1)
        input = x.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(1*2, len(x), 256)).cuda()
        cell_state = Variable(torch.zeros(1*2, len(x), 256)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.gru1 = nn.GRU(input_size=256, hidden_size=256)

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.dropout(self.fc1(x), p=0.5)
        x, hidden = self.gru1(x)
        x = self.fc2(x)
        x = x.permute(1, 0, 2)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def encode_text_batch(text_batch):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)

    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)

    return text_batch_targets, text_batch_targets_lens

def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)

def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word

def decode_predictions(text_batch_logits):

    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new


def compute_loss(text_batch, text_batch_logits):
    criterion = nn.CTCLoss(blank=0)
    text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(device) # [batch_size]
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)
    return loss

num_epochs = 50
lr = 0.001
weight_decay = 1e-3
clip_norm = 5

rnn_hidden_size = 256
crnn = CRNN(num_chars)
crnn.apply(weights_init)
crnn = crnn.to(device)

optimizer = optim.Adam(crnn.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

epoch_losses = []
iteration_losses = []
num_updates_epochs = []

for epoch in range(1, num_epochs + 1):
    epoch_loss_list = []
    num_updates_epoch = 0
    for image_batch, text_batch in train_loader:
        optimizer.zero_grad()
        text_batch_logits = crnn(image_batch.to(device))
        loss = compute_loss(text_batch, text_batch_logits)
        df = pd.DataFrame(columns=['actual', 'prediction'])
        df['actual'] = text_batch
        df['prediction'] = decode_predictions(text_batch_logits.cpu())
        results_train = pd.concat([results_train, df])
        results_train['prediction_corrected'] = results_train['prediction'].apply(correct_prediction)
        accuracy = accuracy_score(results_train['actual'], results_train['prediction_corrected'])
        iteration_loss = loss.item()

        if np.isnan(iteration_loss) or np.isinf(iteration_loss):
            continue

        num_updates_epoch += 1
        iteration_losses.append(iteration_loss)
        epoch_loss_list.append(iteration_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(crnn.parameters(), clip_norm)
        optimizer.step()

    with torch.no_grad():
        for valid_image_batch, valid_text_batch in valid_loader:
            valid_text_batch_logits = crnn(valid_image_batch.to(device))
            valid_text_batch_pred = decode_predictions(valid_text_batch_logits.cpu())
            df = pd.DataFrame(columns=['actual', 'prediction'])
            valid_loss = compute_loss(valid_text_batch, valid_text_batch_logits)
            df['actual'] = valid_text_batch
            df['prediction'] = valid_text_batch_pred
            results_valid['prediction_corrected'] = results_valid['prediction'].apply(correct_prediction)
            valid_accuracy = accuracy_score(results_valid['actual'], results_valid['prediction_corrected'])
            results_valid = pd.concat([results_valid, df])

    results_train,results_valid = results_train.reset_index(drop=True),results_valid.reset_index(drop=True)
    epoch_loss = np.mean(epoch_loss_list)
    print("Epoch:{}    Loss:{}    Accuracy:{}   Valid_Accuracy:{} ".format(epoch, epoch_loss,accuracy,valid_accuracy))
    epoch_losses.append(epoch_loss)
    num_updates_epochs.append(num_updates_epoch)
    lr_scheduler.step(epoch_loss)
    # TensorBoard
    writer.add_scalar('Loss', epoch_loss, epoch)
    writer.add_scalar('Accuracy', accuracy, epoch)


with torch.no_grad():
    for image_batch, text_batch in test_loader:
        text_batch_logits = crnn(image_batch.to(device))
        text_batch_pred = decode_predictions(text_batch_logits.cpu())
        df = pd.DataFrame(columns=['actual', 'prediction'])
        df['actual'] = text_batch
        df['prediction'] = text_batch_pred
        results_test = pd.concat([results_test, df])
results_test = results_test.reset_index(drop=True)

results_test['prediction_corrected'] = results_test['prediction'].apply(correct_prediction)
test_accuracy = accuracy_score(results_test['actual'], results_test['prediction_corrected'])
print(test_accuracy)


mistakes_df = results_test[results_test['actual'] != results_test['prediction_corrected']]
print(mistakes_df)
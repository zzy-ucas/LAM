import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from utils import to_var
import numpy as np


# ========================================Knowing When to Look========================================
class AttentiveCNN(nn.Module):
    # def __init__(self):
    def __init__(self, model_path, num_classes):
        super(AttentiveCNN, self).__init__()
        if model_path is "":
            modules = list(models.vgg16(pretrained=True).features)[:29]
            # modules = list(models.vgg19_bn(pretrained=True).features)[:50]
            # modules = list(models.vgg16(pretrained=True).features)[:26]
            # modules = list(models.vgg16_bn(pretrained=True).features)[:41]
            vgg_conv = nn.Sequential(*modules)  # last conv feature

            self.vgg_conv = vgg_conv
            self.classifier = nn.Sequential(nn.Linear(512 * 14 * 14, num_classes))

            self.init_weights()
        else:
            # vgg = models.vgg11_bn(pretrained=False)
            # vgg.features = torch.nn.Sequential(*(list(vgg.features)[:40]))
            # vgg.classifier = torch.nn.Sequential(torch.nn.Linear(512 * 7 * 7, num_classes))
            vgg = models.vgg16_bn(pretrained=False)
            vgg.features = torch.nn.Sequential(*(list(vgg.features)[:40]))
            vgg.classifier = torch.nn.Sequential(torch.nn.Linear(512 * 14 * 14, num_classes))
            # vgg = models.vgg19_bn(pretrained=False)
            # vgg.features = torch.nn.Sequential(*(list(vgg.features)[:50]))
            # vgg.classifier = torch.nn.Sequential(torch.nn.Linear(512 * 14 * 14, num_classes))

            # vgg.load_state_dict(torch.load(model_path))
            vgg.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0)))

            self.vgg_conv = vgg.features
            self.classifier = vgg.classifier

    def init_weights(self):
        """Initialize the weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in')
                m.bias.data.fill_(0)

    def forward(self, images):
        """
        Input: images
        Output: fea_conv=[v_1, ..., v_n], v_g
        """
        # Last conv layer feature map
        fea_conv = self.vgg_conv(images)  # (50L, 512L, 14L, 14L)

        # fea_conv = [ v_1, v_2, ..., v_196 ]
        feature = fea_conv.view(fea_conv.size(0), fea_conv.size(1), -1).transpose(1, 2)
        # print('###', fea_conv.size())
        prob = self.classifier(fea_conv.view(fea_conv.size(0), -1))

        return feature, prob


class Decoder(nn.Module):
    def __init__(self, vis_dim, vis_num, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout_ratio=0.5):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.vis_num = vis_num
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.lstm_cell = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # attention
        self.att_vw = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        self.att_hw = nn.Linear(self.hidden_dim, self.vis_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(self.vis_dim, 1, bias=False)

        self.att_vw_2 = nn.Linear(embed_dim, self.vis_dim, bias=False)
        ###
        # self.lam_2 = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        ###
        self._init_weight()

    def _init_weight(self):
        init.xavier_uniform_(self.fc_out.weight)
        init.xavier_uniform_(self.att_vw.weight)
        init.xavier_uniform_(self.att_vw_2.weight)
        init.xavier_uniform_(self.att_hw.weight)
        init.xavier_uniform_(self.att_w.weight)

    # def _attention_layer(self, features, hiddens):
    def _attention_layer(self, features, hiddens, features_2, pattern):
        """
        :param features:  batch_size  * 196 * 512
        :param hiddens:  batch_size * hidden_dim
        :param features_2:  batch_size * 49 * 512
        :return:
        """
        att_fea = self.att_vw(features)
        # print('### features:', features.size())  # (batch, 196L, 512L)
        # print('### att_fea:', att_fea.size())  # (batch, 196L, 512L)

        # print('### features:', type(features))
        # print('### features_2:', type(features_2))
        # print('### features:', features.size())
        # print('### features_2:', features_2.size())  # (batch, 1L, 512L)
        ###
        # att_fea_2 = self.lam_2(self.att_vw_2(features_2))  # (batch, 1L, 512L)
        ###
        if pattern != 'baseline':
            att_fea_2 = self.att_vw_2(features_2)  # (batch, 1L, 512L)
        # print('### att_fea_2:', att_fea_2.size())  # (batch, 1L, 512L)

        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # print('### hiddens:', hiddens.size())
        # print('### att_h:', att_h.size())  # (batch, 1L, 512L)
        # N-1-D
        ###
        if pattern == 'baseline':
            att_full = F.relu((att_fea + att_h + self.att_bias.view(1, -1, 1)))
        else:
            att_full = F.relu((att_fea + att_h + self.att_bias.view(1, -1, 1) + att_fea_2))
        # att_full = F.relu((att_fea + att_h + self.att_bias.view(1, -1, 1)))

        att_out = self.att_w(att_full).squeeze(2)
        # print('### att_full:', att_full.size())  # (batch, 196L, 512L)
        # print('### att_out:', att_out.size())  # (batch, 196L)
        alpha = F.softmax(att_out, 1)
        # print('### alpha', alpha.size())  # (batch, 196L)
        # N-L
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        # print('### context', context.size())  # (batch, 512L)
        ###
        # att_fea_2 = att_fea_2.squeeze(1)
        # if use_class:
        #     context = context + att_fea_2
        ###
        return context, alpha

    # def forward(self, features, captions, masks, lengths):
    def forward(self, features, preds, captions, lengths, pattern):
        """
        :param features:
        :param captions: batch_size * time_steps
        :return:
        """
        batch_size, time_step = captions.data.shape
        vocab_size = self.vocab_size
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out
        # print(type(captions))
        # print(type(preds))
        # print(captions.size())  # (batch,)
        # print(preds.size())  # (batch,)
        word_embeddings = embed(captions)
        # print(word_embeddings.size())  # (512L, 24L, 512L)
        preds = embed(preds)  # (512L, 512L)

        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings
        feas = torch.mean(features, 1)  # batch_size * 512
        h0, c0 = self.get_start_states(batch_size)

        predicts = to_var(torch.zeros(batch_size, time_step, vocab_size))

        for step in xrange(time_step):
            batch_size = sum(i >= step for i in lengths)
            if step != 0:
                feas, alpha = attention_layer(
                    # features[:batch_size, :], h0[:batch_size, :])
                    features[:batch_size, :], h0[:batch_size, :], preds[:batch_size], pattern)
            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            # print('### feas_size:', feas.size())  # (batch, 512L)
            # print('### words:', words.size())  # (batch, 512L)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
            predicts[:batch_size, step, :] = outputs

        return predicts

    # def sample(self, feature, max_len=20):
    def sample(self, feature, preds, pattern, max_len=20):
        # greedy sample
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = feature.size(0)

        sampled_ids = []
        alphas = [0]

        words = embed(to_var(torch.ones(batch_size, 1).long())).squeeze(1)
        # print(type(feature))
        # print(type(feature_2))
        # feature_2 = feature_2.unsqueeze(1)
        # print(feature.size())
        # print(feature_2.size())
        # print(feature_2)
        if pattern != 'baseline':
            preds = embed(preds)
        feas = torch.mean(feature, 1)  # convert to batch_size*512
        h0, c0 = self.get_start_states(batch_size)

        for step in xrange(max_len):
            if step != 0:
                # feas, alpha = attend(feature, h0)
                feas, alpha = attend(feature, h0, preds, pattern)
                alphas.append(alpha)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            outputs = fc_out(h0)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.unsqueeze(1))
            words = embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze(), alphas

    def get_start_states(self, batch_size):
        hidden_dim = self.hidden_dim
        h0 = to_var(torch.zeros(batch_size, hidden_dim))
        c0 = to_var(torch.zeros(batch_size, hidden_dim))
        return h0, c0


# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder(nn.Module):
    def __init__(self, args, vocab_size):
        # def __init__(self, args, vocab_size, gpuid):
        super(Encoder2Decoder, self).__init__()
        # Parse Args
        model_path = args.vgg_model_path
        num_classes = args.num_classes
        vis_dim = args.vis_dim
        vis_num = args.vis_num
        embed_dim = args.embed_dim
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        dropout_ratio = args.dropout_ratio
        # gpu = int(args.gpu)

        # Image CNN encoder and Adaptive Attention Decoder
        # self.encoder = AttentiveCNN()
        self.encoder = AttentiveCNN(model_path, num_classes)
        self.decoder = Decoder(vis_dim, vis_num, embed_dim, hidden_dim, vocab_size, num_layers, dropout_ratio)

    def forward(self, images, captions, lengths, pattern):
        # def forward(self, images, captions, lengths, use_class):
        # features = self.encoder(images)
        features, probs = self.encoder(images)
        # print('### fea_conv:', feature.size())  # (batch, 196L, 512L)
        # print('### prob:', prob.size())  # (batch, 21L)
        preds = torch.max(probs.data, 1)[1].unsqueeze(1)
        preds += 4
        preds = to_var(preds)
        # img_ids = to_var(torch.LongTensor(img_ids)//100).unsqueeze(1)

        # print('### predict:', predict.size())  # (batch, 1L)

        # Language Modeling on word prediction
        scores = self.decoder(features, preds, captions, lengths, pattern)

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)

        return packed_scores

    # Caption generator
    def sampler(self, images, max_len=20):
        """
        Samples captions for given image features (Greedy search).
        """

        # Data parallelism if multiple GPUs
        fea_conv = self.encoder(images)

        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1).cuda())
        else:
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1))

        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []
        Beta = []

        # Initial hidden states
        states = None

        for i in range(max_len):
            scores, states, atten_weights, beta = self.decoder(fea_conv, captions, states)
            # scores, states, atten_weights, beta = self.decoder(fea_conv, prob, captions, states)
            predicted = scores.max(2)[1]  # argmax
            captions = predicted

            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append(captions)
            attention.append(atten_weights)
            Beta.append(beta)

        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_ids = torch.cat(sampled_ids, dim=1)
        attention = torch.cat(attention, dim=1)
        Beta = torch.cat(Beta, dim=1)

        return sampled_ids, attention, Beta


if __name__ == '__main__':
    # encoder = models.vgg16()
    # print(encoder)
    params_size_cnn = 3 * 3 * (
            (224 * 224 * 3 * 64) +
            (224 * 224 * 64 * 64) +

            (112 * 112 * 64 * 128) +
            (112 * 112 * 128 * 128) +

            (56 * 56 * 128 * 256) +
            (56 * 56 * 256 * 256) +
            (56 * 56 * 256 * 256) +

            (28 * 28 * 256 * 512) +
            (28 * 28 * 512 * 512) +
            (28 * 28 * 512 * 512)

        # (14 * 14 * 512 * 512) +
        # (14 * 14 * 512 * 512) +
        # (14 * 14 * 512 * 512)
    )
    params_size_lam = 4 * 512 * 512 + 28 * 28 * 512 * 7 + 196
    params_size_lstm = 4 * (512 + 512) * 512 + 512
    print(params_size_lstm)

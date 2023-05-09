from __future__ import unicode_literals, print_function, division
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import math
import random
from seq2seq.models import EncoderRNN, AttnDecoderRNN
from seq2seq.processing import prepareData, tensorFromSentence, tensorsFromPair, MAX_LENGTH
from seq2seq.tokens import SOS_token, EOS_token
import torch
import torch.nn as nn
from torch import optim
import time


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length: int, teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=encoder.device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=decoder.device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def trainIters(
        pairs, input_lang, output_lang,
        encoder, decoder, n_iters: int, teacher_forcing_ratio: float,
        print_every=1000, plot_every=100, learning_rate=0.01
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [
        tensorsFromPair(pair=random.choice(pairs), input_lang=input_lang, output_lang=output_lang,
                        device=encoder.device)
        for i in range(n_iters)
    ]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(
            input_tensor=input_tensor, target_tensor=target_tensor,
            encoder=encoder, decoder=decoder,
            encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
            criterion=criterion,
            max_length=MAX_LENGTH
        )
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device=encoder.device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=encoder.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=decoder.device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang=input_lang, output_lang=output_lang,
                                            encoder=encoder, decoder=decoder, sentence=pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(sentence, encoder, decoder, input_lang, output_lang):
    output_words, attentions = evaluate(input_lang=input_lang, output_lang=output_lang,
                                        encoder=encoder, decoder=decoder, sentence=sentence)
    print('input =', sentence)
    print('output =', ' '.join(output_words))
    showAttention(sentence, output_words, attentions)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device=device) \
        .to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, device=device,
                                   max_length=MAX_LENGTH) \
        .to(device)

    trainIters(
        pairs=pairs, input_lang=input_lang, output_lang=output_lang,
        encoder=encoder1, decoder=attn_decoder1,
        teacher_forcing_ratio=0.5,
        n_iters=75000, print_every=5000
    )

    evaluateRandomly(input_lang=input_lang, output_lang=output_lang,
                     pairs=pairs, encoder=encoder1, decoder=attn_decoder1)

    eval_args = dict(encoder=encoder1, decoder=attn_decoder1, input_lang=input_lang, output_lang=output_lang)
    output_words, attentions = evaluate(sentence="je suis trop froid .", **eval_args)

    plt.matshow(attentions.numpy())

    evaluateAndShowAttention("elle a cinq ans de moins que moi .", **eval_args)

    evaluateAndShowAttention("elle est trop petit .", **eval_args)

    evaluateAndShowAttention("je ne crains pas de mourir .", **eval_args)

    evaluateAndShowAttention("c est un jeune directeur plein de talent .", **eval_args)

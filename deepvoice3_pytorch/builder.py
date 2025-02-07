import torch
from torch import nn

from deepvoice3_pytorch import MultiSpeakerTTSModel, AttentionSeq2Seq, MultispeakerSeq2seq


def deepvoice3(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
               n_speakers=1, speaker_embed_dim=16, padding_idx=0,
               dropout=(1 - 0.95), kernel_size=5,
               encoder_channels=128,
               num_encoder_layer=7,
               decoder_channels=256,
               num_decoder_layer=4,
               attention_hidden=128,
               converter_channels=256,
               num_converter_layer=5,
               query_position_rate=1.0,
               key_position_rate=1.29,
               position_weight=1.0,
               use_memory_mask=False,
               trainable_positional_encodings=False,
               force_monotonic_attention=True,
               use_decoder_state_for_postnet_input=True,
               max_positions=512,
               embedding_weight_std=0.1,
               speaker_embedding_weight_std=0.01,
               freeze_embedding=False,
               window_ahead=3,
               window_backward=1,
               world_upsample = 1,
               sp_fft_size=1025,
               training_type='seq2seq'
               ):
    """Build deepvoice3
    """
    from deepvoice3_pytorch.deepvoice3 import Encoder, Decoder, LinearConverter, WorldConverter

    # Seq2seq
    h = encoder_channels  # hidden dim (channels)
    k = kernel_size   # kernel size
    encoder = Encoder(
        n_vocab, embed_dim, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        embedding_weight_std=embedding_weight_std,
        # (channels, kernel_size, dilation)
        convolutions=[(h, k, 1),]*num_encoder_layer,
    )

    h = decoder_channels
    k = kernel_size
    att_hid = attention_hidden
    decoder = Decoder(
        embed_dim, attention_hidden=att_hid, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        preattention=[(mel_dim*r,h//2),(h//2,h)],
        convolutions=[(h, k, 1),]*num_decoder_layer,
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        position_weight=position_weight,
        use_memory_mask=use_memory_mask,
        window_ahead=window_ahead,
        window_backward=window_backward,
    )

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    if training_type == 'seq2seq':
        scale_speaker_embed = num_encoder_layer + 2 + num_decoder_layer * 2 + 2 #TODO:なくても良いかもなので確認
        model = MultispeakerSeq2seq(
            seq2seq, padding_idx=padding_idx,
            mel_dim=mel_dim,
            n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
            trainable_positional_encodings=trainable_positional_encodings,
            use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
            speaker_embedding_weight_std=speaker_embedding_weight_std,
            freeze_embedding=freeze_embedding, scale_speaker_embed=scale_speaker_embed
        )
        model.training_type = training_type
        return model

    # Post net
    if use_decoder_state_for_postnet_input:
        in_dim = h
    else:
        in_dim = mel_dim
    h = converter_channels
    k = kernel_size

    #Linear or world parameter
    if training_type == 'linear':
        converter = LinearConverter(
            n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
            in_dim=in_dim, out_dim=linear_dim, dropout=dropout, r=r,
            convolutions=[(h,k,1),]*num_converter_layer
        )
    elif training_type == 'world':
        converter = WorldConverter(
            n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
            in_dim=in_dim, out_dim=sp_fft_size, dropout=dropout, r=r, time_upsampling=world_upsample,
            convolutions=[(h, k, 1), ] * num_converter_layer
        )

    scale_speaker_embed = num_encoder_layer + 2 + num_decoder_layer * 2 + 2 + num_converter_layer + 1

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
        speaker_embedding_weight_std=speaker_embedding_weight_std,
        freeze_embedding=freeze_embedding, scale_speaker_embed=scale_speaker_embed)
    model.training_type = training_type

    return model


def nyanko(n_vocab, embed_dim=128, mel_dim=80, linear_dim=513, r=1,
           downsample_step=4,
           n_speakers=1, speaker_embed_dim=16, padding_idx=0,
           dropout=(1 - 0.95), kernel_size=3,
           encoder_channels=256,
           decoder_channels=256,
           converter_channels=512,
           query_position_rate=1.0,
           key_position_rate=1.29,
           use_memory_mask=False,
           trainable_positional_encodings=False,
           force_monotonic_attention=True,
           use_decoder_state_for_postnet_input=False,
           max_positions=512, embedding_weight_std=0.01,
           speaker_embedding_weight_std=0.01,
           freeze_embedding=False,
           window_ahead=3,
           window_backward=1,
           key_projection=False,
           value_projection=False,
           ):
    from deepvoice3_pytorch.nyanko import Encoder, Decoder, Converter
    assert encoder_channels == decoder_channels

    if n_speakers != 1:
        raise ValueError("Multi-speaker is not supported")
    if not (downsample_step == 4 and r == 1):
        raise ValueError("Not supported. You need to change hardcoded parameters")

    # Seq2seq
    encoder = Encoder(
        n_vocab, embed_dim, channels=encoder_channels, kernel_size=kernel_size,
        padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, embedding_weight_std=embedding_weight_std,
    )

    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, channels=decoder_channels,
        kernel_size=kernel_size, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask,
        window_ahead=window_ahead,
        window_backward=window_backward,
        key_projection=key_projection,
        value_projection=value_projection,
    )

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    if use_decoder_state_for_postnet_input:
        in_dim = decoder_channels // r
    else:
        in_dim = mel_dim

    converter = Converter(
        in_dim=in_dim, out_dim=linear_dim, channels=converter_channels,
        kernel_size=kernel_size, dropout=dropout)

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
        speaker_embedding_weight_std=speaker_embedding_weight_std,
        freeze_embedding=freeze_embedding)

    return model


def deepvoice3_multispeaker(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
                            downsample_step=1,
                            n_speakers=1, speaker_embed_dim=16, padding_idx=0,
                            dropout=(1 - 0.95), kernel_size=5,
                            encoder_channels=128,
                            decoder_channels=256,
                            converter_channels=256,
                            query_position_rate=1.0,
                            key_position_rate=1.29,
                            use_memory_mask=False,
                            trainable_positional_encodings=False,
                            force_monotonic_attention=True,
                            use_decoder_state_for_postnet_input=True,
                            max_positions=512,
                            embedding_weight_std=0.1,
                            speaker_embedding_weight_std=0.01,
                            freeze_embedding=False,
                            window_ahead=3,
                            window_backward=1,
                            key_projection=True,
                            value_projection=True,
                            ):
    """Build multi-speaker deepvoice3
    """
    from deepvoice3_pytorch.deepvoice3 import Encoder, Decoder, Converter

    time_upsampling = max(downsample_step // r, 1)

    # Seq2seq
    h = encoder_channels  # hidden dim (channels)
    k = kernel_size   # kernel size
    encoder = Encoder(
        n_vocab, embed_dim, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        embedding_weight_std=embedding_weight_std,
        # (channels, kernel_size, dilation)
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3)],
    )

    h = decoder_channels
    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        preattention=[(h, k, 1)],
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1)],
        attention=[True, False, False, False, False],
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask,
        window_ahead=window_ahead,
        window_backward=window_backward,
        key_projection=key_projection,
        value_projection=value_projection,
    )

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    # Post net
    if use_decoder_state_for_postnet_input:
        in_dim = h // r
    else:
        in_dim = mel_dim
    h = converter_channels
    converter = Converter(
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
        time_upsampling=time_upsampling,
        convolutions=[(h, k, 1), (h, k, 3), (2 * h, k, 1), (2 * h, k, 3)],
    )

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
        speaker_embedding_weight_std=speaker_embedding_weight_std,
        freeze_embedding=freeze_embedding)

    return model

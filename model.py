from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

def get_model(vocab_size=25000):
    config_encoder = BertConfig()
    config_decoder = BertConfig()

    config_encoder.vocab_size = vocab_size
    config_decoder.vocab_size = vocab_size

    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    config.decoder_start_token_id = 2
    config.pad_token_id = 0
    config.eos_token_id = 3

    model = EncoderDecoderModel(config=config)
    
    return model
import abliterator

model = "meta-llama/Meta-Llama-3-70B-Instruct"  # the huggingface or path to the model you're interested in loading in
dataset = [abliterator.get_harmful_instructions(), abliterator.get_harmless_instructions()] # datasets to be used for caching and testing, split by harmful/harmless
device = 'cuda'                             # optional: defaults to cuda
n_devices = None                            # optional: when set to None, defaults to `device.cuda.device_count`
cache_fname = 'my_cached_point.pth'         # optional: if you need to save where you left off, you can use `save_activations(filename)` which will write out a file. This is how you load that back in.
activation_layers = None                    # optional: defaults to ['resid_pre', 'resid_mid', 'resid_post'] which are the residual streams. Setting to None will cache ALL activation layer types
chat_template = None                        # optional: defaults to Llama-3 instruction template. You can use a format string e.g. ("<system>{instruction}<end><assistant>") or a custom class with format function -- it just needs an '.format(instruction="")` function. See abliterator.ChatTemplate for a very basic structure.
negative_toks = [4250]                      # optional, but highly recommended: ' cannot' in Llama's tokenizer. Tokens you don't want to be seeing. Defaults to my preset for Llama-3 models
positive_toks = [23371, 40914]              # optional, but highly recommended: ' Sure' and 'Sure' in Llama's tokenizer. Tokens you want to be seeing, basically. Defaults to my preset for Llama-3 models

my_model = abliterator.ModelAbliterator(
  model,
  dataset,
  device='cuda',
  n_devices=None,
  cache_fname=None,
  activation_layers=['resid_pre', 'resid_post', 'attn_out', 'mlp_out'],
  chat_template="<system>\n{instruction}<end><assistant>",
  positive_toks=positive_toks,
  negative_toks=negative_toks
)

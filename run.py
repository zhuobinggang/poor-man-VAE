import mnist

def run():
  m = mnist.My_VAE_V2(10)
  mnist.train(m, 10)
  mnist.test(m) # will output dd.png
  return m

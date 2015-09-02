function h=sigmoid(a)
  h=1./(1+exp(-a));
  %h=(exp(a)-exp(-a))./(exp(a)+exp(-a));
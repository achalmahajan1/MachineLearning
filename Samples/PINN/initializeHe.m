function weights = initializeHe(sz,numIn)

weights = randn(sz,'single') * sqrt(2/numIn);% Single precision elements
weights = dlarray(weights);% deep learning array for custom training loops (for more info: doc dlarray)

end
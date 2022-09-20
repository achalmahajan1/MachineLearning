function parameter = initializeZeros(sz)
%The function initializeZeros takes as input the size of the learnable parameters sz, 
% and returns the parameters as a dlarray object with underlying type 'single'.
parameter = zeros(sz,'single');
parameter = dlarray(parameter);

end
function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we
%                       want to compute CE(targets, y).
%        frac_correct : (scalar) Fraction of inputs classified correctly.

% TODO: Finish this function
N = size(targets,1);
ce = -sum(targets.*log(y)+(1-targets).*log(1-y))/N;
frac_correct_count = 0;
[targets y];
for i = 1:size(y,1)
	if (targets(i)==0 && y(i)<0.5) || (targets(i)==1 && y(i)>=0.5)
		frac_correct_count = frac_correct_count+1;
	end
end
frac_correct = frac_correct_count/size(y,1);
    
end

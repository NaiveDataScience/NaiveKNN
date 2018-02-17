load quantum.mat
[n,d] = size(X);

% Split into training and validation set
perm = randperm(n);
Xvalid = X(n/2+1:end,:);
yvalid = y(n/2+1:end);
X = X(1:n/2,:);
y = y(1:n/2);

n = n/2;
lambda = 1/n;
model = svmAvg_2(X,y,lambda,25*n);

% count = 0
% for i = 1:size(yvalid,1)
%	if (yvalid(i)>0.5 && y_expected(i)==1) || (yvalid(i)<0.5 && y_expected(i)==0)
%		count += 1;
%	end
% end

% count/size(yvalid,1)
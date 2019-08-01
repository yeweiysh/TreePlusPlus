function [accuracy] = TenFoldCvSvm(K,y)
% K = kernel matrix (n*n)
% y = vector of labels (n*1)
% cv = number of folds in cross-validation

cv=10;
ResultsTenFold=zeros(cv,1);
svmc = 10 .^ (0:1:3);

[n,~]=size(K);
r = randperm(n);
PermK = K(r,r);
Permy=y(r);
clear K; clear y;

Perc80 = ceil(n * 0.8);
Perc90 = ceil(n * 0.9);
fs = n - Perc90;

% cross-validation loop
for k=1:cv
    Seq=[k*fs+1:n,1:(k-1)*fs,(k-1)*fs+1:k*fs];
    NewK=PermK(Seq,Seq);Newy=Permy(Seq);
    Ktr=NewK(1:Perc80,1:Perc80);ytr=Newy(1:Perc80);ntr=length(ytr);
    Kte=NewK(Perc80+1:Perc90,1:Perc80);yte=Newy(Perc80+1:Perc90);nte=length(yte);
    
    Results=zeros(1,length(svmc));
    for i=1:length(svmc)
        model=svmtrain(ytr, [(1:ntr)' Ktr], ['-s 0 -c ', num2str(svmc(i)), ' -t 4 -q 1']);
        [~,acc, ~] = svmpredict(yte, [(1:nte)' Kte], model);%% svm classifier
        %fprintf('The classification accuracy is %f\n',acc(1));
        Results(i)=acc(1);
    end
    clear Ktr; clear Kte;
    % choose optimal svmc
    [~,optimalc]= max(fliplr(Results)); % we perfer large svmc
    optimalc =length(svmc)+1 - optimalc; 
    % train on 90% with optimal c, predict on 10% (from 91% to 100%)
    Ktr90=NewK(1:Perc90,1:Perc90); ytr90=Newy(1:Perc90); ntr90=length(ytr90);
    Kte10=NewK(Perc90+1:n,1:Perc90);yte10=Newy(Perc90+1:n);nte10=length(yte10);
    model=svmtrain(ytr90, [(1:ntr90)' Ktr90], ['-s 0 -c ', num2str(svmc(optimalc)), ' -t 4 -q 1']);
    [~,acc, ~] = svmpredict(yte10, [(1:nte10)' Kte10], model);%% svm classifier
%     fprintf('The %d-fold classification accuracy is %f\n',k,acc(1));
    ResultsTenFold(k)=acc(1);
end
accuracy=mean(ResultsTenFold);

end


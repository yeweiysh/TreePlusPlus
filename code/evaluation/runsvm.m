name={'BZR','BZR_MD','COX2','COX2_MD','DHFR','DHFR_MD',...
      'ER_MD','ENZYMES','KKI','Mutagenicity',...
    'NCI1','PTC_FR','PTC_MM','PTC_MR','PROTEINS_full'};

for dataset=1:15
    disp(['run ' name{dataset}])
    filename=['./kernels/' name{dataset} '_kernel.mat'];
    load(filename);
    filename=['../datasets/' name{dataset} '.mat'];
    load(filename);
    kernel = normalize_kernel(kernel);
    NumExp=10;
    accuracy = zeros(NumExp,1);
    for i=1:NumExp
        accuracy(i)=TenFoldCvSvm(kernel,label);
    end
    MeanAcc=mean(accuracy);StdErr=std(accuracy);
    fprintf('The mean accuracy is %f, and the standard error is %f\n',MeanAcc, StdErr)
end

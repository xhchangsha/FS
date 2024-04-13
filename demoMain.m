opt.lamda1 = 0.03;
opt.lamda2 = 3;
opt.L = 0.3;

algorithm = {'FS_optimize'} ;
dataname = 'Yale15.mat';%dataset
load(char(dataname));
X = NormalizeFea(X,0);%normalization
data = [X, Y];
opt.algorithm = algorithm{1};

t = 1;
percent = [0.02 0.1 0.15 0.2 0.25 0.3 0.35 0.4];
per_num = length(percent);

while t<=10%10 times of 10-fold cross-validation
    indices=crossvalind('Kfold',size(data,1),10);
      
    for k=1:10
        testnum=(indices==k);
        trainnum=~testnum;
        X_test=X(testnum==1,:);
        X_train=X(trainnum==1,:);
        Y_test=Y(testnum==1,:);
        Y_train=Y(trainnum==1,:);
       
        [n, d] = size(X_train);
        W = FS_optimize(X_train',Y_train',opt);
        [T_Weight, T_sorted_features] = sort(W,'descend');
        for i=1:per_num
            p=percent(i);        
            Num_SelectFeaLY = floor(p*d);       
            SelectFeaIdx = T_sorted_features(1:Num_SelectFeaLY);
            X_trainwF = X_train(:,SelectFeaIdx);
            X_testwF = X_test(:,SelectFeaIdx);
            [svm_aCC(i,k)] =LIB_SVM(Y_train,Y_test,X_trainwF,X_testwF);%The acc of the k-th fold validation when selecting percent[i] features in the t-th iteration of 10-fold cross-validation
        end;
        
    end
    for i=1:per_num
        e_svm_aCC=svm_aCC(i,:);
        svm_epercent_epoch_aCC(i,t) = mean(e_svm_aCC);%mean of the t-th iteration of 10-fold cross-validation when selecting percent[i] features
    end;
    t=t+1;
end
for i=1:per_num
    p=percent(i);
    epercent_svm_aCC=svm_epercent_epoch_aCC(i,:);
    epercent_svm_meanaCC = mean(epercent_svm_aCC); %mean of the 10 times 10-fold cross-validation when selecting percent[i] features      
    save(['result\',char(dataname),'_svm_',char(opt.algorithm),'_best_result','_',num2str(p),'_',num2str(epercent_svm_meanaCC),'.mat'],'epercent_svm_aCC','epercent_svm_meanaCC','indices','opt','W');%save result
end

    


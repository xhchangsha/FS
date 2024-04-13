%%%
%INPUT:
%
% X is a d by n matrix, where n is the number of samples and d the number of features.
% Y is row vector with class labels (e.g., 0, 1)
%
% OUTPUT:
%
% w is the score of each feature
%%%

function [w]= FS_optimize(X,Y,opt)
[d,n]=size(X);
w=rand(d,1);%initialize w
m=unique(Y); 
c=length(m);%number of classes
t=1;
L=opt.L;%1/L is equivalent to the learning rate.

objv_new=10^12;
objv=10^12+1;

while ~(objv-objv_new<0.1||t>30)
    [objv,dF]=F_dF_function(X, Y, w, c, m, opt.lamda1, opt.lamda2);%the value of the objective function and derivative of the convex term with respect to w 
    swap=objv_new;
    objv_new=objv;
    objv=swap;
    z=w-dF./L;
    pre_w=w;
    for i=1:d
        if opt.lamda1/L<z(i)
            w(i)=z(i)-opt.lamda1/L;
        elseif abs(z(i))<=opt.lamda1/L
            w(i)=0;
        elseif z(i)<-opt.lamda1/L
            w(i)=z(i)+opt.lamda1/L;
        end
    end
    %objv_hat=objv+dF'*(w-pre_w)+L/2*sum((w-pre_w).*(w-pre_w)); 
    if objv_new>objv&&objv_new<10000
        w=pre_w;
        break 
    end
    t=t+1;

end 


function KL = KL_function(w,pinv2,e1,U)
KL = 1/2*trace(w'*(e1+U)*w/pinv2);

function dKL = dKL_function(w,pinv2,e1,e2,U)%the kl divergence with respect to w
E=diag(ones(size(w,1),1));
dKL = (-e2*w./pinv2*w'+E)*(e1+U)*w./pinv2;

function [objv, dF]= F_dF_function(X,Y,w,c,m,lamda1, lamda2)
dF=0;
cost=0;%cost function
for i=1:c
    mKL=0;
    mdKL=0;
    X1=X(:,Y==m(i));
    [s1,s2]=size(X1);
    noise=0.1*randn(s1,s2);
    X2=X1+noise;
    u1=mean(X1,2);
    e1=cov(X1');
    u2=mean(X2,2);
    e2=cov(X2');
    pinv1=w'*e1*w;
    pinv2=w'*e2*w;
    U12=(u1-u2)*(u1-u2)';
    for j=i+1:c
        if j~=i
            X3=X(:,Y==m(j));
            u3=mean(X3,2);
            e3=cov(X3');
            pinv3=w'*e3*w;
            U13=(u1-u3)*(u1-u3)';
            mdKL=(dKL_function(w,pinv2,e1,e2,U12)-dKL_function(w,pinv3,e1,e3,U13));          
            mKL=KL_function(w,pinv2,e1,U12)-KL_function(w,pinv3,e1,U13);
        end
        dF=dF+mdKL/(1+exp(-mKL));
        cost=cost+log(1+exp(mKL));
    end   
end
dF=dF+2*lamda2.*w;
objv=cost+lamda1*norm(w,1)+lamda2*norm(w,2)^2;
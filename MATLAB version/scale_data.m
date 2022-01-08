function [scaled_train_x1,scaled_train_x2] = scale_data(train_x1,train_x2)
    mu1 = mean(train_x1)';
    sigma1 = cov(train_x1);
    mu2 = mean(train_x2)';
    sigma2 = cov(train_x2);
%     [V1,D1] = eig(sigma1);
%     [V2,D2] = eig(sigma2);
    [U1,S1,V1] = svd(sigma1);
    [U2,S2,V2] = svd(sigma2);
    p = mu1 - mu2;
%     size(p)
%     size(train_x1)
    scaled_train_x1 = zeros(length(train_x1),2);
%     train_x2
    scaled_train_x1(:,1) = train_x1*p;
%     scaled_train_x1(:,2) = train_x1*V1(:,end);
    scaled_train_x1(:,2) = train_x1*U1(:,1);
    scaled_train_x2 = zeros(length(train_x2),2);
    scaled_train_x2(:,1) = train_x2*p;
%     scaled_train_x2(:,2) = train_x2*V2(:,end);
    scaled_train_x2(:,2) = train_x2*U2(:,1);
end
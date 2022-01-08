clc;
clear all;
data = readtable('balance-scale.csv');
disp('balance')
x1_count = 0;
x2_count = 0;
x3_count = 0;
for i=1:height(data)
    class = char(data.class(i));
    if class=='B'
        x1_count = x1_count +1;
    end
    if class=='R'
        x2_count = x2_count +1;
    end
    if class=='L'
        x3_count = x3_count +1;
    end
end
features = 4;
x1 = zeros(x1_count,features);
x2 = zeros(x2_count,features);
x3 = zeros(x3_count,features);
% all = zeros(150,4);
x1_count = 1;
x2_count = 1;
x3_count = 1;
for i=1:height(data)
    class = char(data.class(i));
    if class=='B'
        x= table2array(data(i,{'f1','f2','f3','f4'}));
        x1(x1_count,:) = x;
        x1_count = x1_count + 1;
    end
    if class=='R'
        x= table2array(data(i,{'f1','f2','f3','f4'}));
        x2(x2_count,:) = x;
        x2_count = x2_count + 1;
    end
    if class=='L'
        x= table2array(data(i,{'f1','f2','f3','f4'}));
        x3(x3_count,:) = x;
        x3_count = x3_count + 1;
    end
end
%%
% [x2,x3] = scale_data(x2,x3);
% figure(1)
% plotmatrix(x1,x3(1:49,:))
% figure(2)
% plotmatrix(x2,x3)
% hold on
% scatter(x1(:,1),x1(:,2))
% figure(3)
% plotmatrix(x2(1:48,:),x3)



%%
% x2 = vertcat(x2,x3);
x1 = x2;
x2 = x3;


error = [];
for j=1:10
    y1 = zeros(length(x1),1);
    y2 = ones(length(x2),1);
    y1_indices = crossvalind('Kfold',y1,10);
    y2_indices = crossvalind('Kfold',y2,10);
    loss = 0;
    for i=1:10
        test = (y1_indices == i); 
        train = ~test;
        train_x1 = x1(train,:);
        test_x1 = x1(test,:);

        test = (y2_indices == i); 
        train = ~test;
        train_x2 = x2(train,:);
        test_x2 = x2(test,:);
        [scaled_train_x1,scaled_train_x2] = scale_data(train_x1,train_x2);
        
        pso_loss = PSO(scaled_train_x1,scaled_train_x2,test_x1,test_x2,i);
        loss = loss + (pso_loss / sum(length(test_x1),length(test_x2)));
%         input('Press enter')
    end
    error(j)=loss/10;

end
min(error)

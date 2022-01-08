clc;
clear all;
disp('iris');
data = readtable('iris.csv');
x1_count = 0;
x2_count = 0;
x3_count = 0;
for i=1:height(data)
    class = data.class(i);
    if strcmp(class,'Iris-virginica')
        x3_count = x3_count +1;
    end
    if strcmp(class,'Iris-setosa')
        x1_count = x1_count +1;
    end
    if strcmp(class,'Iris-versicolor')
        x2_count = x2_count +1;
    end
end
x1 = zeros(x1_count,4);
x2 = zeros(x2_count,4);
x3 = zeros(x3_count,4);
% all = zeros(150,4);
x1_count = 1;
x2_count = 1;
x3_count = 1;
for i=1:height(data)
    class = data.class(i);
    if strcmp(class,'Iris-virginica')
        x= table2array(data(i,{'sepal_length','sepal_width','petal_length','petal_width'}));
        x3(x3_count,:) = x;
        x3_count = x3_count + 1;
%         all(i,:) = x;
    end
    if strcmp(class,'Iris-setosa')
        x= table2array(data(i,{'sepal_length','sepal_width','petal_length','petal_width'}));
        x1(x1_count,:) = x;
        x1_count = x1_count + 1;
%         all(i,:) = x;
    end
    if strcmp(class,'Iris-versicolor')
        x= table2array(data(i,{'sepal_length','sepal_width','petal_length','petal_width'}));
        x2(x2_count,:) = x;
        x2_count = x2_count + 1;
%         all(i,:) = x;
    end
end
%%
% test_x1 = x1(length(x1)-9:length(x1),:);
% train_x1 = x1(1:length(x1)-10,:);
% test_x2 = x2(length(x2)-9:length(x2),:);
% train_x2 = x2(1:length(x2)-10,:);
% test_x3 = x3(length(x3)-9:length(x3),:);
% train_x3 = x3(1:length(x3)-10,:);
% figure(1)
% scatter(x3(:,1),x3(:,2))
% hold on
% scatter(x2(:,1),x2(:,2))
% train_x2 = vertcat(train_x2,train_x3);
% test_x2 = vertcat(test_x2,test_x3);
%%
x2 = vertcat(x2,x3);


error = [];
for j=1:10
    loss = 0;
    y1 = zeros(length(x1),1);
    y2 = ones(length(x2),1);
    y1_indices = crossvalind('Kfold',y1,10);
    y2_indices = crossvalind('Kfold',y2,10);
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
%         scaled_train_x1= train_x1;
%         scaled_train_x2 =train_x2;
        pso_loss = PSO(scaled_train_x1,scaled_train_x2,test_x1,test_x2,i);
        loss = loss + (pso_loss / sum(length(test_x1),length(test_x2)));
%         input('Press enter')
    end
    error(j) = loss/10;
end
min(error)

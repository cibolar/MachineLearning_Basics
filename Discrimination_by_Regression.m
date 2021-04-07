%                       CEYHUN IBOLAR

%% Reading the csv file from hw02_data_set_images
Data_Set_Images = csvread('hw02_data_set_images.csv');

%Reading the csv file from hw02_data_set_labels
fileID = fopen('hw02_data_set_labels.csv');
Scan_The_Text = textscan(fileID,'%s');
Data_Set_Labels = Scan_The_Text{1,1};
fclose(fileID);

%% Constan values from data points
  k=5;    %number of classes
  [N,D] = size(Data_Set_Images);
  N_training = 125; %number of training labels
  N_testing = N-N_training;% number of testing labels 

%% Dividing each classes first 25 letters for training and last 14 for testing

%A
Training_Data_Set_Images.A = Data_Set_Images(1:25,:);
Training_Data_Set_Labels.A = Data_Set_Labels(1:25,:);
Testing_Data_Set_Images.A  = Data_Set_Images(26:39,:);
Testing_Data_Set_Labels.A = Data_Set_Labels(26:39,:);
%B
Training_Data_Set_Images.B= Data_Set_Images(40:64,:);
Training_Data_Set_Labels.B = Data_Set_Labels(40:64,:);
Testing_Data_Set_Images.B  = Data_Set_Images(65:78,:);
Testing_Data_Set_Labels.B = Data_Set_Labels(65:78,:);
%C
Training_Data_Set_Images.C = Data_Set_Images(79:103,:);
Training_Data_Set_Labels.C = Data_Set_Labels(79:103,:);
Testing_Data_Set_Images.C  = Data_Set_Images(104:117,:);
Testing_Data_Set_Labels.C = Data_Set_Labels(104:117,:);
%D
Training_Data_Set_Images.D = Data_Set_Images(118:142,:);
Training_Data_Set_Labels.D = Data_Set_Labels(118:142,:);
Testing_Data_Set_Images.D  = Data_Set_Images(143:156,:);
Testing_Data_Set_Labels.D  = Data_Set_Labels(143:156,:);
%E
Training_Data_Set_Images.E = Data_Set_Images(157:181,:);
Training_Data_Set_Labels.E = Data_Set_Labels(157:181,:);
Testing_Data_Set_Images.E  = Data_Set_Images(182:195,:);
Testing_Data_Set_Labels.E = Data_Set_Labels(182:195,:);

X_testing = [Testing_Data_Set_Images.A;Testing_Data_Set_Images.B;Testing_Data_Set_Images.C;Testing_Data_Set_Images.D;Testing_Data_Set_Images.E];
Y_testing = [Testing_Data_Set_Labels.A;Testing_Data_Set_Labels.B;Testing_Data_Set_Labels.C;Testing_Data_Set_Labels.D;Testing_Data_Set_Labels.E];

X_training  = [Training_Data_Set_Images.A;Training_Data_Set_Images.B;Training_Data_Set_Images.C;Training_Data_Set_Images.D;Training_Data_Set_Images.E];
Y_training  = [Training_Data_Set_Labels.A;Training_Data_Set_Labels.B;Training_Data_Set_Labels.C;Training_Data_Set_Labels.D;Training_Data_Set_Labels.E];
% Converting training classes into integers
% Class A -> 1, Class B -> 2, Class C -> 3, Class D -> 4, Class E -> 5
Y_training = str2num(cell2mat(strrep(strrep(strrep(strrep((strrep...
                                      (Y_training,...
                                      '"A"','1')),...
                                      '"B"','2'),...
                                      '"C"','3'),...
                                      '"D"','4'),...
                                      '"E"','5')));
% Converting testing classes into integers
% Class A -> 1, Class B -> 2, Class C -> 3, Class D -> 4, Class E -> 5                                  
Y_testing=  str2num(cell2mat(strrep(strrep(strrep(strrep((strrep...
                                      (Y_testing,...
                                      '"A"','1')),...
                                      '"B"','2'),...
                                      '"C"','3'),...
                                      '"D"','4'),...
                                      '"E"','5')));
                                  
  y = zeros(length(Y_training),k);
  
  j = 1;
  %one hot coding for training data
 for i=1:1:length(Y_training)
     for j=1:1:k
         if(Y_training(i)==j)
             y(i,j)=1;
         else
             y(i,j)=0;
         end
     end
 end
 
 y_training = y;

%% Main Function
eta = 0.001;
epsilon = 0.00001;

a = -0.01;
b = 0.01;
N = length(X_training);

%get a random value in the range of a and b for N values
w_1  = a + (b-a) .* rand(N,1);
w0_1 = a + (b-a) .* rand(1,1);
w_2  = a + (b-a) .* rand(N,1);
w0_2 = a + (b-a) .* rand(1,1);
w_3  = a + (b-a) .* rand(N,1);
w0_3 = a + (b-a) .* rand(1,1);
w_4  = a + (b-a) .* rand(N,1);
w0_4 = a + (b-a) .* rand(1,1);
w_5  = a + (b-a) .* rand(N,1);
w0_5 = a + (b-a) .* rand(1,1);

W = [w_1,w_2,w_3,w_4,w_5];
W0 = [w0_1;w0_2;w0_3;w0_4;w0_5];

iteration = 1;

while 1
     
     y_predicted =  sigmoid(X_training,w_1,w0_1,w_2,w0_2,w_3,w0_3,w_4,w0_4,w_5,w0_5);
       
     w_old   = [w_1,w_2,w_3,w_4,w_5];
     w_old_0 = [w0_1,w0_2,w0_3,w0_4,w0_5];
     
     gradient_w_output = gradient_w (X_training,y_training,y_predicted);
     gradiant_w0_output = gradient_w0 (y_training,y_predicted);
          
     w_1    = w_1   + eta*gradient_w_output(:,1);
     w_2    = w_2   + eta*gradient_w_output(:,2);
     w_3    = w_3   + eta*gradient_w_output(:,3);
     w_4    = w_4   + eta*gradient_w_output(:,4);
     w_5    = w_5   + eta*gradient_w_output(:,5);
     w0_1   = w0_1  + eta*gradiant_w0_output(1)';
     w0_2   = w0_2  + eta*gradiant_w0_output(2)';
     w0_3   = w0_3  + eta*gradiant_w0_output(3)';
     w0_4   = w0_4  + eta*gradiant_w0_output(4)';
     w0_5   = w0_5  + eta*gradiant_w0_output(5)';
     
     Error(iteration) = sum(sum((y_training- y_predicted').^2));
     
     % w_3 is the weight that is calculated the most difficult. For that I
     % took the only weight for stoping the iteration. 
     if(sum(w_3-w_old(:,3))^2  + sum(((w0_3-w_old_0(:,3))^2))) < epsilon
         figure;
         plot(Error,'r','LineWidth',2)
         xlabel('Iteration');
         ylabel('Error');
         title('Error-Iteration Graph (epsilon = 0.00001) ');
         grid('on')
         break;
     end
      
     iteration = iteration+1;     
end

%% Comparing the score values
y_predicted_score = zeros(N_training*k,1);
y_predicted = y_predicted';

% With the max() function, label with max. score is calculated
    for t=1:1:N_training
        [~,label_with_max_score] = max(y_predicted(t,1:k));
        y_predicted_score(t) = label_with_max_score;
    end
 
 confusion_matrix = zeros(5,5);
 for i=1:1:N_training
     c = y_predicted_score(i);
     m = Y_training(i);        
     confusion_matrix(c,m) = confusion_matrix(c,m) + 1;
 end
 

disp("Confusion Matrix For Training Data Points :");
disp(confusion_matrix);

%% Using the TESTING data points with the calculated W,W0 values.

y_predicted_testing = sigmoid(X_testing,w_1,w0_1,w_2,w0_2,w_3,w0_3,w_4,w0_4,w_5,w0_5);

y_predicted_testing_score = zeros(N_testing*k,1);
y_predicted_testing = y_predicted_testing'; 

    for t=1:1:N_testing
        [~,label_with_max_score] = max(y_predicted_testing(t,1:k));
        y_predicted_testing_score(t) = label_with_max_score;
    end
    
     confusion_matrix_testing = zeros(5,5);
     for i=1:1:N_testing
         c = y_predicted_testing_score(i);
         m = Y_testing(i);        
         confusion_matrix_testing(c,m) = confusion_matrix_testing(c,m) + 1;
     end  

disp("Confusion Matrix For Testing Data Points :");
disp(confusion_matrix_testing);

%% UI ( It is just related with making a figure for confusion matrix when running the MATLAB)
f = figure;
% f2 = figure(2);
uit = uitable(f);
uit2 = uitable(f);

% d = {confusion_matrix};
uit.Data = confusion_matrix;
uit.Position = [10 250 545 125];
uit.ColumnName = {'y_train    1','2','3','4','5'};
uit.ColumnEditable = true;
uit.RowName = {'y_predicted   1','2','3','4','5'};
txt_title = uicontrol('Style', 'text', 'Position', [225 375 110 40], 'String', 'Confusion Matrix');

uit2.Data = confusion_matrix_testing;
uit2.Position = [10 100 545 125];
uit2.ColumnName = {'y_test   1','2','3','4','5'};
uit2.RowName = {'y_predicted   1','2','3','4','5'};

%% Sigmoid function
function    sigmoid_output = sigmoid    (x,...
                                         w_1,w0_1,...
                                         w_2,w0_2,...
                                         w_3,w0_3,...
                                         w_4,w0_4,...
                                         w_5,w0_5)
      
      sigmoid_output_1 = (1)./(1+exp(-(w_1'*x'+w0_1)));
      sigmoid_output_2 = (1)./(1+exp(-(w_2'*x'+w0_2)));
      sigmoid_output_3 = (1)./(1+exp(-(w_3'*x'+w0_3)));
      sigmoid_output_4 = (1)./(1+exp(-(w_4'*x'+w0_4)));
      sigmoid_output_5 = (1)./(1+exp(-(w_5'*x'+w0_5)));
      sigmoid_output = [sigmoid_output_1;sigmoid_output_2;sigmoid_output_3;sigmoid_output_4;sigmoid_output_5];

end

%% Gradient w function
function gradient_w_output = gradient_w(X,y_truth,y_predicted)

         gradient_w_output =  X'*((y_truth-y_predicted').*y_predicted'.*(1-y_predicted'));
                  
 end

%% Gradient w0 function
function gradiant_w0_output = gradient_w0(y_truth,y_predicted)                                          
                                         
         gradiant_w0_output = sum(((y_truth-y_predicted').*y_predicted'.*(1-y_predicted')));
        
end



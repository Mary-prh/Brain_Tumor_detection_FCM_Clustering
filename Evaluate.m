function [Accuracy, Sensitivity,Dice, Jaccard, Specitivity,TP,TN,FP,FN,Precision] = Evaluate(A, B)

% A is the ground truth, B is the segmented result.
% MCC - Matthews correlation coefficient
% Note: Sensitivity = Recall
% TP - true positive, FP - false positive,
% TN - true negative, FN - false negative


X = A;
Y = B;

% Evaluate TP, TN, FP, FN
sumindex = X + Y;
TP = length(find(sumindex == 2));
TN = length(find(sumindex == 0));
substractindex = X - Y;
FP = length(find(substractindex == -1));
FN = length(find(substractindex == 1));
Accuracy = (TP+TN)/(FN+FP+TP+TN);
Sensitivity = TP/(TP+FN);
Precision = TP/(TP+FP);
Dice = 2*TP/(2*TP+FP+FN);
Jaccard = Dice/(2-Dice);
Specitivity = TN/(TN+FP);
Ppv = TP/(TP+FP);
Npv = TN/(TN+FN);
end
close all
clear 
%% reading data
data = load('88.mat'); 
Data = data.cjdata;
image = Data.image;
image = im2uint8(image);
tumor_mask = Data.tumorMask;
figure(1);
subplot(1,2,1),
imshow(image,[])
subplot(1,2,2),
imshow(tumor_mask,[])


%% Section A: Pre-Processing
% Step A.1: Contrast Enhancement
I_adjust = imadjust(image);
figure(2);
imshow(I_adjust,[])

% Step A.2: Skull Removal
% A.2.1: thresholding
level = graythresh(I_adjust);
BW = imbinarize(I_adjust,level);
figure(3);
imshow(BW,[])
title(['Binary image by threshold level of ', num2str(level)])

% A.2.2: morphological refinements
% Closing
SE1= strel('disk',40);
Im_close = imclose(BW, SE1);
figure(4);
imshow(Im_close,[])
title('Gaps are covered by closing operator with disk SE of 20 ')

%Erosion
SE2= strel('disk',35);
Im_erode = imerode(Im_close,SE2);
figure(5);
imshow(Im_erode,[])
title('Mask is shrinked by erosion operator with disk SE of 35 ')

% A.2.3: Masking the skull part
I_adjust(~Im_erode)=0;
figure(6);
imshow(I_adjust,[])
title('Pre-processed and masked image')
figure(7);
imhist(I_adjust)
title('Histogram of the Pre-processed Image')
xlabel('Image intensities')
ylabel('Number of pixels')

%% Section B: FCM Clustering

% Step B.1 : convert image to vector 1-by-(image size)
I = im2double(I_adjust); % data must be scalar data not integers,
[R1,R2] = size(I); % size is extracted for rebuilding the image in the final stage
I_Vector = I(:)'; % each pixel intensity will become a sample for clustering

% Step B.2: Define variables
N = size(I_Vector,2); % number of samples (R1 * R2)
k = 6; % number of clusters
Nepoch = 50; % number of iteration to converge
Mu= rand(k,N); % initializing 'k' membership functions with random numbers for each pixel
centers= zeros(1,k); % intializing 'k' cluster centers
m = 2; % degree of fuzziness

% Step B.3: FCM Formula implementation
tic
for iter= 1:Nepoch  % in iterations, centers and membership functions update
    % Step B.3.1: update centers
    num =((Mu.^ m) * I_Vector')'; % numinator of the center formula (Eq. 2 in the report)
    denum = sum(Mu.^m,2)'; % denuminator of the center formula (Eq. 2 in the report) which is summation of all MF
    centers = num./(denum+eps); % center formula (Eq. 2 in the report),
    % ... eps is a small value to avoid getting NAN if denuminator was zero
    
    % Step B.3.2: update membership functions
    for j=1:k % 'k': Number of clusters, the distance of each sample to all clusters is callculated, here
        cj= centers(:,j); % in each iteration one center comes
        reptCj= repmat(cj,1,N); % the center repeats for all samples
        dis(j,:)=  sqrt(((reptCj-I_Vector).^2)); % distance of each sample with the following centers is calculated
    end
    
    for j=1:k % 'k': Number of clusters, the distance of each sample to each cluster is callculated, here
        disj= dis(j,:); % distance of each sample with the  'j'th center is calculated
        Dj= repmat(disj,k,1); % and we repeat it for all clusters so that we divide it by 'dis'
        denumM=(Dj./dis).^(2/(m-1)); % ... this is the denuminator in denuminator of equation 3
        % ... that the distance of each sample with the  'j'th center
        % is divided by the distance to all centers
        Dnum=sum(denumM); % and we sum them all together (all clusters)
        Mu(j,:)= 1./(Dnum+eps); % this is the equation 3 in the report
    end
end
toc
C = centers;
U = Mu;

% Step B.4: finalize clustering
[mx_val,labels]= max(U); % U is the membership function. lebels (column address)
% ... are given to the max of membership function for each pixel

Clustered_Image = zeros(size(labels)); % creating a vector by the size of the labels
for i=1:k
    indx= find(labels==i); % in each iteration the index of each label is extracted and ...
    Clustered_Image(indx)= centers(i); % ... the corresponding center is taken
end

% Step B.5: turning the vector to image
Clustered_Image = reshape(Clustered_Image,R1,R2);
figure (7);
imshow(Clustered_Image,[]);
title('The result of clustering the image with FCM method')
%% Section C: Post-Processing
% Step D.1: tumor extraction
mx = max(Clustered_Image(:)); % tumor has the highest intensity levels,
Tumor_area = (Clustered_Image == mx); % so we only keep the cluster with the highest center intensity value
figure (8);
imshow(Tumor_area)
title('Potential Tumor Area')

%% Section D: 
% Step D.2: Removing Noise and keeping Tumor only
Im_1 = Tumor_area;
CC = bwconncomp(Im_1) % connecte components with connectivity of 8 are analyzed
numPixels = cellfun(@numel,CC.PixelIdxList); %the number of  pixels in the compinents are extracted
[s1,s2] = size(numPixels); % in the case that image has only one component, we should consider that as the tumor
if s2 > 1
    % Determine which is the largest component in the image and erase it (set all the pixels to 0).
    [biggest,idx] = max(numPixels);
    Im_1(CC.PixelIdxList{idx}) = 0; % the index of the largest component is ignored to create a mask from noise
    figure (9);
    imshow(Im_1)
    title('Noise Mask')
    
    Tumor = imsubtract(Tumor_area,Im_1); % noise is subtracted from the tumor area image to get the tumor
    figure (10);
    imshow(Tumor)
    title('Detected Tumor Mask')
else
    Tumor = Tumor_area;
    figure (9);
    imshow(Tumor)
    title('Detected Tumor area')
end

% Step D.3: filling gaps in the tumor region
strl= strel('disk',4); % structuring element with disk of radius 4
Final_Tumor = imclose(Tumor,strl); % closing operation to fill the gaps in tumor region
figure (11);
imshow(Final_Tumor)
title('Final Tumor Mask')
%% performance
A = rescale(tumor_mask);
B = Final_Tumor;

[Accuracy, Sensitivity,Dice, Jaccard, Specitivity,TP,TN,FP,FN,Precision] = Evaluate(A, B);


figure (12);
imshowpair(A,B,'montage')
title(['Ground Truth (Left) and Detected Tumor (Right), Accuracy = ', num2str(Accuracy), ' , Sensitivity = ' , num2str(Sensitivity)])



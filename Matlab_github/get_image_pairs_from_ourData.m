%% registration of the Real-world images and cropping into image pairs
% use loop_get_img_pairs_from_outdata.m to run it in loops
% clear;
% clc;
% close all;
% addpath(genpath('SR_Mean'));
% cd C:\Rui_Onedrive\OneDrive\2021Spring\Super-Registration
addpath(genpath('SR_ourdata_newCam'));
addpath('./RANSAC-example');
run('vlfeat-0.9.21/toolbox/vl_setup');

%% X4
% scale = 4; % should be changed by the txt file
% target F1
% fileID = fopen('target_F1_C640_X4.txt','r');
% fileID = fopen('target_F1_C1300_X4.txt','r');
% fileID = fopen('target_F1_C4112_X4.txt','r');
% save_folder = "SR_mean/target_F1_C640_X4";
% save_folder = "SR_mean/target_F1_C1300_X4";
% save_folder = "SR_mean/target_F1_C4112_X4";

% target F2
% fileID = fopen('target_F2_C640_X4.txt','r');
% fileID = fopen('target_F2_C1300_X4.txt','r');
% fileID = fopen('target_F2_C4112_X4.txt','r');
% save_folder = "SR_mean/target_F2_C640_X4";
% save_folder = "SR_mean/target_F2_C1300_X4";
% save_folder = "SR_mean/target_F2_C4112_X4";
% target F2 newCam
% fileID = fopen('target_F2_X4.txt','r');
% fileID = fopen('target_F2_C1300_X4.txt','r');
% fileID = fopen('target_F2_C4112_X4.txt','r');
% save_folder = "SR_ourdata_newCam/target_F2_X4";
% save_folder = "SR_mean/target_F2_C1300_X4";
% save_folder = "SR_mean/target_F2_C4112_X4";

% target J1
% fileID = fopen('target_J1_C640_X4.txt','r');
% fileID = fopen('target_J1_C1300_X4.txt','r');
% fileID = fopen('target_J1_C4112_X4.txt','r');
% save_folder = "SR_mean/target_J1_C640_X4";
% save_folder = "SR_mean/target_J1_C1300_X4";
% save_folder = "SR_mean/target_J1_C4112_X4";
% target J1 newCam
% fileID = fopen('target_J1_X4.txt','r');
% save_folder = "SR_ourdata_newCam/target_J1_X4";
% target K1
% fileID = fopen('target_K1_C640_X4.txt','r');
% fileID = fopen('target_K1_C1300_X4.txt','r');
% fileID = fopen('target_K1_C4112_X4.txt','r');
% save_folder = "SR_mean/target_K1_C640_X4";
% save_folder = "SR_mean/target_K1_C1300_X4";
% save_folder = "SR_mean/target_K1_C4112_X4";

% target S1
% fileID = fopen('target_S1_C640_X4.txt','r');
% fileID = fopen('target_S1_C1300_X4.txt','r');
% fileID = fopen('target_S1_C4112_X4.txt','r');
% save_folder = "SR_mean/target_S1_C640_X4";
% save_folder = "SR_mean/target_S1_C1300_X4";
% save_folder = "SR_mean/target_S1_C4112_X4";
% target S1 newCam
% fileID = fopen('target_S1_X4.txt','r');
% save_folder = "SR_ourdata_newCam/target_S1_X4";
%% X2
scale = 2; % should be changed by the txt file
% target F1
% fileID = fopen('target_F1_C640_X2.txt','r');
% fileID = fopen('target_F1_C1300_X2.txt','r');
% fileID = fopen('target_F1_C4112_X2.txt','r');
% save_folder = "SR_mean/target_F1_C640_X2";
% save_folder = "SR_mean/target_F1_C1300_X2";
% save_folder = "SR_mean/target_F1_C4112_X2";

% target F2
% fileID = fopen('target_F2_C640_X2.txt','r');
% fileID = fopen('target_F2_C1300_X2.txt','r');
% fileID = fopen('target_F2_C4112_X2.txt','r');
% save_folder = "SR_mean/target_F2_C640_X2";
% save_folder = "SR_mean/target_F2_C1300_X2";
% save_folder = "SR_mean/target_F2_C4112_X2";
% % target F2 newCam
% fileID = fopen('SR_ourdata_newCam/target_F2_X2.txt','r');
% save_folder = "SR_ourdata_newCam/target_F2_X2";
% target J1
% fileID = fopen('target_J1_C640_X2.txt','r');
% fileID = fopen('target_J1_C1300_X2.txt','r');
% fileID = fopen('target_J1_C4112_X2.txt','r');
% save_folder = "SR_mean/target_J1_C640_X2";
% save_folder = "SR_mean/target_J1_C1300_X2";
% save_folder = "SR_mean/target_J1_C4112_X2";
% target J1 newCam
fileID = fopen('SR_ourdata_newCam/target_J1_X2.txt','r');
save_folder = "SR_ourdata_newCam/target_J1_X2";
% target K1
% fileID = fopen('target_K1_C640_X2.txt','r');
% fileID = fopen('target_K1_C1300_X2.txt','r');
% fileID = fopen('target_K1_C4112_X2.txt','r');
% save_folder = "SR_mean/target_K1_C640_X2";
% save_folder = "SR_mean/target_K1_C1300_X2";
% save_folder = "SR_mean/target_K1_C4112_X2";

% target S1
% fileID = fopen('target_S1_C640_X2.txt','r');
% fileID = fopen('target_S1_C1300_X2.txt','r');
% fileID = fopen('target_S1_C4112_X2.txt','r');
% save_folder = "SR_mean/target_S1_C640_X2";
% save_folder = "SR_mean/target_S1_C1300_X2";
% save_folder = "SR_mean/target_S1_C4112_X2";
% % target S1 newCam
% fileID = fopen('SR_ourdata_newCam/target_S1_X2.txt','r');
% save_folder = "SR_ourdata_newCam/target_S1_X2";
%% begin to create
% read the txt file that containing the paths of target image pairs
line = fgetl(fileID); 
% % try to run from somewhere in the middle
cnt = 0;
while line ~= -1
    cnt = cnt + 1;
%     if cnt <= 262
%         line = fgetl(fileID); 
%         continue;
%     end
    p = strsplit(line,' ');
    path1 = p{1};
    path2 = p{2};
%     % for mat file
%     img1 = load(path1).avg_img;
%     img2 = load(path2).avg_img;
    % for tif file
    img1 = double(imread(fullfile(path1)));
    img2 = double(imread(fullfile(path2)));
    % resize HR into the same scale as LR
    img2_small = imresize(img2, 1/scale);
    % pre-matching
    c = normxcorr2(img2_small, img1);
    [~,I] = sort(c(:));
    % if the similarity <= 0.9, we don't want this pair
    if c(I(end)) <= 0.9
        line = fgetl(fileID); 
        continue;
    end
    % finding three best candidates that at least 100 pixels away from each
    % other
    dis_away = 100;
    [ypeak1,xpeak1] =find(c==c(I(end)));
    [ypeak2,xpeak2] =find(c==c(I(end-1)));
    cnt = 2;
    while (ypeak2-ypeak1)^2 + (xpeak2-xpeak1)^2 <= dis_away^2
        [ypeak2,xpeak2] =find(c==c(I(end-cnt)));
        cnt = cnt + 1;
    end
    [ypeak3,xpeak3] =find(c==c(I(end-cnt)));
    while (ypeak3-ypeak1)^2 + (xpeak3-xpeak1)^2 <= dis_away^2 || (ypeak3-ypeak2)^2 + (xpeak3-xpeak2)^2 <= dis_away^2
        [ypeak3,xpeak3] =find(c==c(I(end-cnt)));
        cnt = cnt + 1;
    end    
    dy1 = ypeak1-size(img2_small,1);
    dx1 = xpeak1-size(img2_small,2);
    dy2 = ypeak2-size(img2_small,1);
    dx2 = xpeak2-size(img2_small,2);
    dy3 = ypeak3-size(img2_small,1);
    dx3 = xpeak3-size(img2_small,2);
%     [ypeak,xpeak] = find(c==max(c(:)));
%     dy = ypeak-size(img2_small,1);
%     dx = xpeak-size(img2_small,2);
%     figure;
%     imshow(uint8([img1,img2]));
    if contains(path1,"C4112") || contains(path2,"C4112") 
        FirstOctave = 1;
        peaks_thresh = 5;
    else
        FirstOctave = 0;
        peaks_thresh = 2;
    end
%     peaks_thresh = 7; % for only target_F2
    if contains(path1,"K1")     
        peaks_thresh = 0; % for only target_J1 C640 X4, all of target_K1 
    end    
%     FirstOctave = 1;
    range = 10;
    Match_threshold = 15;
    if contains(path1,"J1")
        Match_threshold = 15;
    end
    [~, ~, Matched_Pts1] = SIFTnNewmatch_multiDetectors(img1, img2_small, -dx1, -dy1, 'SIFT', range, Match_threshold, FirstOctave, peaks_thresh);
    [~, ~, Matched_Pts2] = SIFTnNewmatch_multiDetectors(img1, img2_small, -dx2, -dy2, 'SIFT', range, Match_threshold, FirstOctave, peaks_thresh);
    [~, ~, Matched_Pts3] = SIFTnNewmatch_multiDetectors(img1, img2_small, -dx3, -dy3, 'SIFT', range, Match_threshold, FirstOctave, peaks_thresh);
    if size(Matched_Pts1,1) >= size(Matched_Pts2,1) && size(Matched_Pts1,1) >= size(Matched_Pts3,1)
        Matched_Pts = Matched_Pts1;
    elseif size(Matched_Pts2,1) > size(Matched_Pts1,1) && size(Matched_Pts2,1) >= size(Matched_Pts3,1)
        Matched_Pts = Matched_Pts2;
    elseif size(Matched_Pts3,1) > size(Matched_Pts1,1) && size(Matched_Pts3,1) > size(Matched_Pts2,1)
        Matched_Pts = Matched_Pts3;
    end
%     Matched_Pts = Matched_Pts1;% for only C640 target_F2

%     [dxnew, dynew, Matched_Pts] = SIFTnNewmatch_multiDetectors(img1, img2_small, -dx, -dy, 'SIFT', 10, 10, FirstOctave);
%     Matched_Pts = SIFTnNewmatch_compare(img1, img2, 'SURF');
    if size(Matched_Pts,1)>=4 
        % get the real points in HR and show matches
%         figure;
%         showMatches(uint8(img1), uint8(img2), Matched_Pts(:,1:2), Matched_Pts(:,3:4).*scale-(scale-1)/2);
        
        pts1 = [Matched_Pts(:,[2,1]),ones(size(Matched_Pts,1),1)]';
        % convert scaled img2 into origin
        pts2 = [Matched_Pts(:,[4,3]).*scale-(scale-1)/2,ones(size(Matched_Pts,1),1)]';
%         pts2 = [Matched_Pts(:,[4,3]).*4,ones(size(Matched_Pts,1),1)]';
        TransM = get_affine_knownScale(scale, pts1, pts2);
        % crop the LR based on TransM
        fprintf("processing %s and %s \n", path1, path2);
        LR = crop_LR(img1, img2, TransM, scale, "DOG", "SSIM");

        % show matches
        if ~isempty(LR)
            figure;
%             imshow(uint8([imresize(img1(83:202,83:242),4),img2]) )
            imshow(uint8( [imresize(LR,scale),img2] ) );
            title_name = sprintf("showing %s and %s", path1, path2);
            title(title_name, 'Interpreter','none');
            % to save the image pair, get the name from path1 and path2 
%             name = "C640_lens05_001_lens20_001.mat";
            [filepath1,name1,ext1] = fileparts(path1);
            [filepath2,name2,ext2] = fileparts(path2);
            filepath1 = strrep(filepath1,'/','_');
            filepath2 = strrep(filepath2,'/','_');
            save_name1 = "LR_" + string(filepath1) + "_" + name1 + "_" ...
                         + filepath2 + "_" + name2 + ".mat";
            save_name2 = "HR_" + string(filepath1) + "_" + name1 + "_" ...
                         + filepath2 + "_" + name2 + ".mat";
            save_show = "pair_" + string(filepath1) + "_" + name1 + "_" ...
             + filepath2 + "_" + name2 + ".tif";
            if ~exist(save_folder,'dir')
                mkdir(save_folder);
            end
            name = save_folder + "/" + save_name1;
            save(name,'LR');
            name = save_folder + "/" + save_name2;
            save(name,'img2');   
            saveas(gcf,[save_folder + "/" + save_show]);
            close;
        end

    else
        disp("not enough matching points, go for next pair!");
    end
    
    
    
    line = fgetl(fileID);
end
%% notes
% from resize bicubicly
% [1 2 3 4 5 6 7 8] to [1 2]  
% LR 1 represents 2.5 in HR
% LR 2 represents 6.5 in HR
% LR*4-1.5 ?= HR

%% from points to matrix
% solved by function: TransM = get_affine_knownScale(scale, pts1, pts2)
% our case: a11 = a22 = scale
% A = [a11, a12, a13;
%      a23, a22, a23;
%        0,   0,   1;]
% X = [xi yi 1  0  0  0;
%       0  0 0 xi yi  1 ]
% a = [a11; a12; a13; a21; a22; a23]
% X*a = [xi'; yi']
% X*a = x_prime
% a = X \ x_prime

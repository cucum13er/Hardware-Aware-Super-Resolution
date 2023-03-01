function LR = crop_LR(img1, img2, TransM, scale, metric, similarity)
% given the truth that img2 are mostly scale and translation, we can assume
% that the transformed four corners of img2 can be the crop borders of img1
if nargin < 6
    similarity = "SSIM";
end
if nargin < 5
    metric = "DOG";
end
if nargin < 4
    scale = 4;
end
if metric == "DOG"
    if similarity == "SSIM"
        fprintf("evaluation is to use DOG filter with 2 octaves, 3 levels, and ssim as the similarity\n");
    elseif similarity == "L1"
        fprintf("evaluation is to use DOG filter with 2 octave, 3 level, and mean L1 norm as the similarity\n");
    else
        error("wrong similarity method, please choose SSIM or L1!");
    end
elseif metric == "Gradient"
    if similarity == "SSIM"
        fprintf("evaluation is to use gradient |dx|+|dy|, and ssim as the similarity\n");   
    elseif similarity == "L1"
        fprintf("evaluation is to use gradient |dx|+|dy|, and L1 norm as the similarity\n");   
    else
        error("wrong similarity method, please choose SSIM or L1!");
    end
elseif metric == "Original"
    if similarity == "SSIM"
        fprintf("evaluation is original imgs with SSIM as the similarity\n");
    elseif similarity == "L1"
        fprintf("evaluation is original imgs with L1 norm as the similarity\n");  
    else
        error("wrong similarity method, please choose SSIM or L1!");
    end    
else
    error("wrong metric, please choose DOG, Gradient, or Original!");
end
[h,w] = size(img2);
% pt2 = TransM * pt1 -> pt1 = TransM^-1 * pt2
topleft2 = [1,1];
topright2 = [1,w];
bottomleft2 = [h,1];
bottomright2 = [h,w];

topleft1 = TransM^-1 * [topleft2,1]';
topright1 = TransM^-1 * [topright2,1]';
bottomleft1 = TransM^-1 * [bottomleft2,1]';
bottomright1 = TransM^-1 * [bottomright2,1]';

crop_top = min(topleft1(1), topright1(1));
crop_left = min(topleft1(2), bottomleft1(2));
crop_bottom = max(bottomleft1(1), bottomright1(1));
crop_right = max(topright1(2), bottomright1(2));
%% method 2: give some candidates and calculate the RMSE and SSIM values
% top and bottom
top = floor(crop_top);
bottom = ceil(crop_bottom);
% left and right
left = floor(crop_left);
right = ceil(crop_right);

fprintf("bottom and top gap is %d, right and left gap is %d \n", bottom-top+1, right-left+1);

if (bottom-top+1) - size(img2,1)/scale == 0

elseif ismember( (bottom-top+1) - size(img2,1)/scale,  [1,2] )
    top = top + 1;
elseif ismember( (bottom-top+1) - size(img2,1)/scale,  [3,4] )
    top = top + 2;
elseif ismember( (bottom-top+1) - size(img2,1)/scale,  [5,6] )
    top = top + 3;    
else
    LR = [];
    disp("top and bottom not aligned well!");
    return    
end

if (right-left+1) - size(img2,2)/scale == 0
    
elseif ismember( (right-left+1) - size(img2,2)/scale, [1,2])
    left = left + 1;
elseif ismember( (right-left+1) - size(img2,2)/scale, [3,4])
    left = left + 2;
elseif ismember( (right-left+1) - size(img2,2)/scale, [5,6])
    left = left + 3;    
else
    LR = [];
    disp("left and right not aligned well!");
    return    
end

tops = top-5:top+5;
lefts = left-5:left+5;

% if the top and left out of range
if tops(end) <= 0 || lefts(end) <= 0
    disp("left or top is less than 1!");
    LR = [];
    return
elseif ~isempty(find(tops,1)) || ~isempty(find(lefts,1))
    idx_top = find(tops==1);
    if ~isempty(idx_top)
        tops = tops(idx_top:end);
    end
    idx_left = find(lefts==1);
    if ~isempty(idx_left)
        lefts = lefts(idx_left:end);
    end
end
% if the bottom or right correspondences are out of range
if tops(1)+h/scale-1 > h || lefts(1)+w/scale-1 > w
    disp("bottom or right is greater than maximum boarder!");
    LR = [];
    return
elseif ~isempty(find(tops, h-h/scale+1)) || ~isempty( find(lefts, w-w/scale+1))
    idx_bottom = find(tops==h-h/scale+1);
    if ~isempty(idx_bottom)
        tops = tops(1:idx_bottom);
    end
    idx_right = find(lefts==w-w/scale+1);
    if ~isempty(idx_right)
        lefts = lefts(1:idx_right);
    end    
end
%% Find the best candidate from 11*11 candidates
% metric = "DOG";
% similarity = "SSIM";
SSIM = 0;
L1_norm = inf;
for i = 1:length(tops)
    for j = 1:length(lefts)
        LR_tmp = img1(tops(i):tops(i)+h/scale-1, lefts(j):lefts(j)+w/scale-1);
        HR = imresize(img2,1/scale);
        if similarity == "SSIM"
            ssim_tmp = cal_similarity(HR, LR_tmp, metric, similarity);
            if ssim_tmp > SSIM
                SSIM = ssim_tmp;
                top = tops(i);
                left = lefts(j);
            end
        elseif similarity == "L1"
            norm_tmp = cal_similarity(HR, LR_tmp, metric, similarity);
            if norm_tmp < L1_norm
                L1_norm = norm_tmp;
                top = tops(i);
                left = lefts(j);
            end
        else
            error("wrong similarity method, please choose SSIM or L1!")
        end        

        
    end
end

%% conclusion
LR = img1(top:top+h/scale-1, left:left+w/scale-1,:);
end
% help function: calculate the similarity between the given HR and LR 
% images using different metrci and simialrity measurements
function eval_res = cal_similarity(HR, LR, metric, similarity)
if nargin < 4
    similarity = "SSIM";
end
if nargin < 3
    metric = "DOG";
end

if metric == "DOG"
    % set the octaves and levels of DOG filters
    octave = 2; level=3;
    dogHR = create_DOG_imgs(HR, octave, level);
    dogLR = create_DOG_imgs(LR, octave, level);
    if similarity == "SSIM"
        SSIM = [];
        for i = 1:length(dogHR)
            for j = 1:size(dogHR{1},3)
                tmp_ssim = ssim(dogHR{i}(2:end-1, 2:end-1, j), dogLR{i}(2:end-1, 2:end-1, j) );
                SSIM = [SSIM, tmp_ssim];
            end
        end
%         fprintf("evaluation is to DOG filter with %d octave, %d level, and ssim as the similarity\n", octave, level);
        eval_res = mean(SSIM);        
    
    elseif similarity == "L1"
        L1norm = [];
        for i = 1:length(dogHR)
            for j = 1:size(dogHR{1},3)
                errorM = dogHR{i}(2:end-1, 2:end-1, j) - dogLR{i}(2:end-1, 2:end-1, j);
                tmp_L1 = norm(errorM(:), 1)/length(errorM(:));
                L1norm = [L1norm, tmp_L1];
            end
        end
%         fprintf("evaluation is to DOG filter with %d octave, %d level, and mean L1 norm as the similarity\n", octave, level);
        eval_res = mean(L1norm); 
    else
        error("wrong similarity method, please choose SSIM or L1!");
    end    

elseif metric == "Gradient"
    [Xh,Yh] = gradient(HR);
    gradH = (abs(Xh)+abs(Yh));
    [Xl,Yl] = gradient(LR_tmp);
    gradL = (abs(Xl)+abs(Yl));
    if similarity == "SSIM"
%         fprintf("evaluation is to use gradient |dx|+|dy|, and ssim as the similarity\n");      
        eval_res = ssim(gradL,gradH);
    end
    if similarity == "L1"
%         fprintf("evaluation is to use gradient |dx|+|dy|, and L1 norm as the similarity\n");      
        errorM = gradL-gradH;
        eval_res = norm(errorM(:), 1);
    end    

elseif metric == "Original"
    if similarity == "SSIM"
%         fprintf("evaluation is original imgs with SSIM as the similarity\n");
        eval_res = ssim(HR, LR);
    elseif similarity == "L1"
%         fprintf("evaluation is original imgs with L1 norm as the similarity\n");
        errorM = HR-LR;
        eval_res = norm(errorM(:), 1);
    end
else 
    error("wrong metric, please choose DOG, Gradient, or Original!");
end

end

%% supplemental material
% % method 1: according to the estimation
% % top and bottom
% top = floor(crop_top);
% bottom = ceil(crop_bottom);
% if abs(abs(bottom-top+1) - size(img2,1)/scale) == 0
%     
% elseif abs(bottom-top+1) - size(img2,1)/scale == 1
%     if top - crop_top >=  bottom - crop_bottom
%         top = top + 1;
%     else
%         bottom = bottom - 1;
%     end
% elseif abs(bottom-top+1) - size(img2,1)/scale == 3
%     if top - crop_top >=  bottom - crop_bottom
%         top = top + 2;
%         bottom = bottom - 1;
%     else
%         top = top + 1;
%         bottom = bottom - 2;
%     end    
% elseif abs(bottom-top+1) - size(img2,1)/scale == -1
%     if top - crop_top >= bottom - crop_bottom
%         bottom = bottom + 1;
%     else
%         top = top - 1;
%     end    
%     
% elseif abs(bottom-top+1) - size(img2,1)/scale == 2
%     top = top + 1;
%     bottom = bottom - 1;
% elseif abs(bottom-top+1) - size(img2,1)/scale == 4
%     top = top + 2;
%     bottom = bottom - 2;    
% % elseif abs(bottom-top) - size(img1,1) == -2
% else
%     LR = [];
%     disp("top and bottom not aligned well!");
%     return
% end
% % left and right
% left = floor(crop_left);
% right = ceil(crop_right);
% if abs(abs(right-left+1) - size(img2,2)/scale) == 0
%     
% elseif abs(abs(right-left+1) - size(img2,2)/scale) == 1
%     if crop_left - left >= right - crop_right
%         left = left + 1;
%     else
%         right = right - 1;
%     end
% elseif abs(abs(right-left+1) - size(img2,2)/scale) == 3
%     if crop_left - left >= right - crop_right
%         left = left + 2;
%         right = right - 1;
%     else
%         right = right - 2;
%         left = left + 1;
%     end    
% elseif abs(abs(right-left+1) - size(img2,2)/scale) == -1
%     if crop_left - left >= right - crop_right
%         right = right + 1;
%     else
%         left = left - 1;
%     end    
%     
% elseif abs(abs(right-left+1) - size(img2,2)/scale) == 2
%     left = left + 1;
%     right = right - 1;
% elseif abs(abs(right-left+1) - size(img2,2)/scale) == 4
%     left = left + 2;
%     right = right - 2;
% % elseif abs(bottom-top) - size(img1,1) == -2
% else
%     LR = [];
%     disp("left and right not aligned well!");
%     return
% end

% %         ssim_tmp = ssim(HR, LR_tmp);
% %         try gradient
%         [Xh,Yh] = gradient(HR);
%         gradH = (abs(Xh)+abs(Yh));
%         [Xl,Yl] = gradient(LR_tmp);
%         gradL = (abs(Xl)+abs(Yl));
%         ssim_tmp = ssim(gradL,gradH);
% %         norm_tmp = norm(gradL - gradH, 1);
% %         norm_tmp = norm(HR - LR_tmp, 1);
% %         if norm_tmp < L1norm

%             imshow([rescale(gradL),rescale(gradH)])
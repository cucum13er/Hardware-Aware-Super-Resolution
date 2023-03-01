% SIFT feature and new matching method

%% Using SIFT to find key blobs
% get panorama using SIFT features
% input: image_store is a imageDatastore data type
function [dxnew, dynew, Matched_Pts] = SIFTnNewmatch_multiDetectors(im1, im2, dx, dy, detector, range, Match_threshold, FirstOctave,peak_thresh)
% tic;
if (nargin < 9)
    peak_thresh = 0;
end
if (nargin < 8)
    FirstOctave = 0;
end
if (nargin < 7)
    Match_threshold = 10;
end
if (nargin < 6)
    range = 30;
end


if size(im1,3) == 3
    im1 = rgb2gray(im1);
    im2 = rgb2gray(im2);
end
if detector == "SIFT"
    im1 = Chg_Single(im1);
    im2 = Chg_Single(im2);
    %% SIFT parameters
    % 'PeakThresh', obtaining fewer features as peak_thresh is increased
    % 'edgethresh', obtaining more features as edge_thresh is increased
    % parameters
    Octaves = 5;
    Levels = 5;
%     FirstOctave = 0;% already put into function parameter, 1 for big img
%     peak_thresh = 0;
    edge_thresh = 500;
    % NormThresh = 0;
    Magnif = 3;
    WindowSize = 2;
    % Frames;
    % Orientations;
    % Verbose;
    [points, features] = vl_sift(im1, 'edgethresh', edge_thresh,...
                                'PeakThresh', peak_thresh,'Levels',Levels,...
                                'Octaves',Octaves,'FirstOctave',FirstOctave,...
                                'Magnif',Magnif,...
                                'WindowSize',WindowSize);
    points = points';
    features = features';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shows features
% figure;
% grayImage = insertShape(uint8(im1),'circle',...
%      [points(:,1:2), points(:,3).*ones(size(points,1),1)],'LineWidth',2,'Color','green');
% imshow(grayImage);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Store points and features for im1
pointsPrevious = points;
featuresPrevious = features;
% Detect and extract SIFT features for im2
[points, features] = vl_sift(im2, 'edgethresh', edge_thresh,...
                            'PeakThresh', peak_thresh,'Levels',Levels,...
                            'Octaves',Octaves,'FirstOctave',FirstOctave,...
                            'Magnif',Magnif,...
                            'WindowSize',WindowSize);
points = points';
features = features';

elseif detector == "SURF"
    % parameters
    MetricThreshold = 1000;
    NumOctaves = 3;
    NumScaleLevels = 4;
    points1 = detectSURFFeatures(im1,'MetricThreshold',MetricThreshold,...
                                 'NumOctaves',NumOctaves,...
                                 'NumScaleLevels',NumScaleLevels);
    [f1,vpts1] = extractFeatures(im1,points1);
    points2 = detectSURFFeatures(im2,'MetricThreshold',MetricThreshold,...
                                 'NumOctaves',NumOctaves,...
                                 'NumScaleLevels',NumScaleLevels);
    [f2,vpts2] = extractFeatures(im2,points2);
    pointsPrevious = vpts1.Location;
    featuresPrevious = f1;
    points = vpts2.Location;
    features = f2;
elseif detector == "ORB"
    points1 = detectORBFeatures(im1, 'ScaleFactor',1.2, 'NumLevels', 8);
    [f1,vpts1] = extractFeatures(im1,points1);
    points2 = detectORBFeatures(im2, 'ScaleFactor',1.2, 'NumLevels', 8);
    [f2,vpts2] = extractFeatures(im2,points2);
    pointsPrevious = vpts1.Location;
    featuresPrevious = f1.Features;
    points = vpts2.Location;
    features = f2.Features;    
% elseif detector == 'Corner'
%     
%     points1 = detectHarrisFeatures(im1);
%     [f1,vpts1] = extractFeatures(im1,points1);
%     points2 = detectHarrisFeatures(im2);
%     [f2,vpts2] = extractFeatures(im2,points2);
%     pointsPrevious = vpts1.Location;
%     featuresPrevious = f1.Features;
%     points = vpts2.Location;
%     features = f2.Features; 
else
    fprintf('please enter a valid detector name.');
    error('No valid detector name found');
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % shows features
% figure;
% grayImage = insertShape(uint8(im2),'circle',...
%      [points(:,1:2), 1*ones(size(points,1),1)],'LineWidth',2,'Color','green');
% imshow(grayImage);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% toc
%% Match those features based on the dx and dy
numFeatures = size(pointsPrevious, 1);
Matched_Pts = [];
%%%%%%%%%%%%%%%%%%%There are problems in random sampling%%%%%%%%%%%%%
% if numFeatures > 3000
%     random_i = datasample(1:numFeatures,3000,"Replace",false);
% else
    random_i = 1:numFeatures;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic;
rows = cell(length(random_i),1);
for i = 1: length(random_i)
    % for each Feature, find out the position in the second image
    x = pointsPrevious(random_i(i),1) + dx;
    y = pointsPrevious(random_i(i),2) + dy;
    % get a searching range from this position
    % find(secondFeature in range)
    [rows{i}, ~] = find((points(:,1)-x).^2+(points(:,2)-y).^2 <= range^2);
%%%%%%%%%%%% old version matching %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     if ~isempty(rows{i})     
%         indexPairs = matchFeatures(featuresPrevious(random_i(i),:), features(rows{i},:), ...
%                                   'MatchThreshold', 3,'Unique', false);
%         % match the feature
%         % save the matching result    
%        
%         if ~isempty(indexPairs)  
%             Matched_Pts = [Matched_Pts; pointsPrevious(random_i(i),1:2), points(rows{i}(indexPairs(2)),1:2)];
%         end
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
% toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%new version mathcing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

featuresPrevious = double(featuresPrevious);
features = double(features);
% Match_threshold = 10; %%%%%%%%%%%%can be revised %%%%%%%%%%%%%%
index = [];
dists = sum(featuresPrevious.^2, 2) + sum(features.^2, 2)' - 2 * (featuresPrevious * features');
for i = 1:length(rows)
    if ~isempty(rows{i})
        [mindis, position] = min(dists(i,rows{i}));
        if mindis <= Match_threshold^2 * 128
            index = [index; [i,rows{i}(position)]];
        end
    end
end
if ~isempty(index)
    Matched_Pts = [pointsPrevious(index(:,1),1:2), points(index(:,2),1:2)];
end
% toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isempty(Matched_Pts)
    dxnew = mean(Matched_Pts(:,3) - Matched_Pts(:,1));
    dynew = mean(Matched_Pts(:,4) - Matched_Pts(:,2));
else
    dxnew = dx;
    dynew = dy;
end


%% update outputs
dxnew = double(dxnew);
dynew = double(dynew);
Matched_Pts = double(Matched_Pts);
Matched_Pts = unique(Matched_Pts,'rows');
%% RANSAC  
% if size(Matched_Pts,1) >= 10 % run RANSAC if over 10 pairs of matching
%     [~, corrPtIdx] = findHomography(Matched_Pts(:,1:2)', Matched_Pts(:,3:4)');
%     Matched_Pts = Matched_Pts(corrPtIdx(1,:),:);
% end
% % shows features
%     grayImage = insertShape((I),'circle',...
%          [points(:,1:2), points(:,3).*ones(size(points,1),1)],'LineWidth',1);
%     figure;
%     imshow(grayImage);
end
%% tool function
function im = Chg_Single(I)
    if size(I,3) == 3
        im = single(rgb2gray(I));
    elseif size(I,3) == 1
        im = single((I));
    end    
end

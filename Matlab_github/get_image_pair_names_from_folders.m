% get txt file containing names of the image pairs
% addpath(genpath('SR_ourdata_newCam'));
root = "SR_ourdata_newCam";
folders = dir(root);
folders = folders(3:end);
isdir = [folders.isdir];
folders(isdir==0) = [];
folders = {folders.name};

for i = 1:length(folders)
    folder_name = fullfile(root,folders{i});   
    folder05 = fullfile(folder_name,"lens05");
    file05 = {dir(folder05).name};
    file05 = file05(3:end);
    folder10 = fullfile(folder_name,"lens10");
    file10 = {dir(folder10).name};
    file10 = file10(3:end);
    folder20 = fullfile(folder_name,"lens20");
    file20 = {dir(folder20).name};
    file20 = file20(3:end);    
    % for 2X
    fileID = fopen(fullfile(root,folders{i}+"_X2.txt"),'w');
    for j = 1:length(file05)
        nameLR = replace(fullfile(folder05, file05{j}), '\', '/');
        for k = 1:length(file10)
            nameHR = replace(fullfile(folder10, file10{k}), '\', '/');
            fprintf(fileID, "%s %s \n",nameLR,nameHR);
        end
    end
    for j = 1:length(file10)
        nameLR = replace(fullfile(folder10, file10{j}), '\', '/');
        for k = 1:length(file20)
            nameHR = replace(fullfile(folder20, file20{k}), '\', '/');
            fprintf(fileID, "%s %s \n",nameLR,nameHR);
        end
    end   
    fclose(fileID);
    % for 4X
    fileID = fopen(fullfile(root,folders{i}+"_X4.txt"),'w');
    for j = 1:length(file05)
        nameLR = replace(fullfile(folder05, file05{j}), '\', '/');
        for k = 1:length(file20)
            nameHR = replace(fullfile(folder20, file20{k}), '\', '/');
            fprintf(fileID, "%s %s \n",nameLR,nameHR);
        end
    end
    fclose(fileID);    
end
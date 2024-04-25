function imgs = create_DOG_imgs(img, octave, level)
    sigma0 = sqrt(2);
    [row, col, ~] = size(img);
    if size(img,3) ~= 1
        img = im2gray(img);
    end
    temp_img = double(img);
    temp_img = kron(img,ones(2));
    temp_img=padarray(temp_img,[1,1],'replicate');
    D=cell(1,octave);
    for i=1:octave
    D(i)=mat2cell(zeros(row*2^(2-i)+2,col*2^(2-i)+2,level),row*2^(2-i)+2,col*2^(2-i)+2,level);
    end
    %create the DoG pyramid.
    for i=1:octave
        temp_D=D{i};
        for j=1:level
            scale=sigma0*sqrt(2)^(1/level)^((i-1)*level+j);
            p=(level)*(i-1);
%             figure(1);
%             subplot(octave,level,p+j);
            f=fspecial('gaussian',[1,floor(6*scale)],scale);
            L1=temp_img;
            if(i==1&&j==1)
            L2=conv2(temp_img,f,'same');
            L2=conv2(L2,f','same');
            temp_D(:,:,j)=L2-L1;
%             imshow(uint8(255 * mat2gray(temp_D(:,:,j))));
            L1=L2;
            else
            L2=conv2(temp_img,f,'same');
            L2=conv2(L2,f','same');
            temp_D(:,:,j)=L2-L1;
            L1=L2;
            if(j==level)
                temp_img=L1(2:end-1,2:end-1);
            end
%             imshow(uint8(255 * mat2gray(temp_D(:,:,j))));
            end
        end
        D{i}=temp_D;
        temp_img=temp_img(1:2:end,1:2:end);
        temp_img=padarray(temp_img,[1,1],'both','replicate');
    end
    imgs = D;
end
%This fuction completes the image stitch with cross-correlation method
%This funtion is for Panorama mosaic, Rename your single image with 'number.tif' which tells the relations of all image 
%like: 0  1  2  3  4
%      5  6  7  8  9
%     10 11 12 13 14
%Input: p,q. you can use this two number to choose which file of the folder you want to stitch
%Output: flag=0,stitch error, x_Diff=404, y_Diff=404
%        flag=1,stitch right, x_Diff=404, y_Diff=404
%Before you start this funtion, change the 'folder_tif' to your own file path 

function [flag, x_Diff, y_Diff] = stitch_correlation(p,q)

folder_tif = 'F:\llc\2D\sr images 40nm\';%file reader
files = dir([folder_tif '*.tif']);
Image1 = imread([folder_tif,files(p).name]);
Image2 = imread([folder_tif,files(q).name]);

%Paramenters and Preprocessing
SubAreaHigh = 300;%Change the subArea size when nessesary
SubAreaWide = 100;        
[row1,colum1] = size(Image1);
[row2,colum2] = size(Image2);
x_Diff = 404;
y_Diff = 404;

mid1 = Image1>80;%binarization
mid2 = Image2>80;
Image1(mid1) = 1;
Image1(~mid1) = 0;
Image2(mid2) = 1;
Image2(~mid2) = 0;

x1 = fix(p/10);%　根据ｐ，ｑ判断两图属于上下／左右拼接
y1 = mod(p,10);
x2 = fix(q/10);
y2 = mod(q,10);

%% 两图拼接
if y1~=y2                                                                  %左右两图拼接
    if y1>y2                                                               %当第一副图在后，交换两图顺序
        ImageMid = Image1;
        Image1 = Image2;
        Image2 = ImageMid;
    end
    
    %get the montage subarea in Image1/sum all the subarea pixels value and choose the maximum as the montage subarea
    t = 0;
    Num = 0;
    for m = 1:50:row1-SubAreaHigh
        MontaArea1 = Image1(m:m+SubAreaHigh, colum1-SubAreaWide:colum1);
        ChoseValue = sum(sum(MontaArea1));
        if ChoseValue > t 
            t = ChoseValue;
            Num = m;
        end
    end
    if t < (1/10)*SubAreaHigh*SubAreaWide%if the two image exist common part,if not,run out
        flag = 0;%拼接标志位，1成功拼接，0拼接失败
        return;
    end
    MontaArea1 = Image1(Num:Num+SubAreaHigh, colum1-SubAreaWide:colum1);

    %Convolution and find the Maximum
    k = 0;
    a = 1;
    Num1 = 0;
    Num2 = 0;
    for y = 1:row2-SubAreaHigh
        for x = 20:100
            MontaArea2 = Image2(y:y+SubAreaHigh, x:x+SubAreaWide);
            Value2 = sum(sum(MontaArea1.*MontaArea2));
            corelation(a) = Value2; 
            if Value2 > k
                k = Value2; 
                Num1 = y;
                Num2 = x;
            end
            a = a+1;
        end 
    end
    y_Diff = Num-Num1;
    x_Diff = colum1-SubAreaWide-Num2+1;

elseif x1~=x2                                                              %上下两图拼接
    if x1>x2                                                               %当第一副图在后，交换两图顺序
        ImageMid = Image1;
        Image1 = Image2;
        Image2 = ImageMid;
    end
    
    %get the montage subarea in Image1/sum all the subarea pixels value and choose the maximum as the montage subarea
    t = 0;
    Num = 0;
    for m = 1:50:colum1-SubAreaHigh
        MontaArea1 = Image1(row1-SubAreaWide:row1, m:m+SubAreaHigh);
        ChoseValue = sum(sum(MontaArea1));
        if ChoseValue > t 
            t = ChoseValue;
            Num = m;
        end
    end
    if t < (1/10)*SubAreaHigh*SubAreaWide                                  %if the two image exist common part,if not,run out
        flag = 0;                                                          %拼接标志位，1成功拼接，0拼接失败
        return;
    end
    MontaArea1 = Image1(row1-SubAreaWide:row1, m:m+SubAreaHigh);

    %Convolution and find the Maximum
    k = 0;
    a = 1;
    Num1 = 0;
    Num2 = 0;
    for y = 1:colum2-SubAreaHigh
        for x = 20:100
            MontaArea2 = Image2(x:x+SubAreaWide, y:y+SubAreaHigh);
            Value2 = sum(sum(MontaArea1.*MontaArea2));
            corelation(a) = Value2; 
            if Value2 > k
                k = Value2; 
                Num1 = y;
                Num2 = x;
            end
            a = a+1;
        end 
    end
    %Give the Result
    x_Diff = Num-Num1;
    y_Diff = colum1-SubAreaWide-Num2+1;
else                                                                       %两图不相邻，程序错误
    print('The two image dose not adjoin');
    return
end
%% 拼接质量测试
% if (Num-Num1)<0
%     Image1_Raw = [ZeroMatrix1;Image1_Raw];
%     Image2_Raw = [Image2_Raw;ZeroMatrix2];
%     imageRef = Image2_Raw(Num1:Num1+SubAreaHigh, Num2-20:Num2+20);%作为质量评价的参考图，从图二中截取
% else
%     Image1_Raw = [Image1_Raw,ZeroMatrix1];
%     Image2_Raw = [ZeroMatrix2;Image2_Raw];
%     imageRef = Image2_Raw(Num1+fillZero:Num1+fillZero+SubAreaHigh, Num2-20:Num2+20);
% end
% MontaImage =[Image1_Raw(:,1:colum1-SubAreaWith),Image2_Raw(:,Num2+1:colum2)];
% if (Num-Num1)<0
%     imageDis = MontaImage(Num+fillZero:Num+fillZero+SubAreaHigh, colum1-SubAreaWith-20:colum1-SubAreaWith+20);%作为质量评价的待评价图，从拼接结果中截取
% else
%     imageDis = MontaImage(Num:Num+SubAreaHigh, colum1-SubAreaWith-20:colum1-SubAreaWith+20);
% end

% %质量评价
% T1 = max(max(imageRef));
% imageRef = 256*(imageRef/T1);
% T2 = max(max(imageDis));
% imageDis = 256*(imageDis/T2);
% FeatureSIM(imageRef, imageDis);
 flag = 1;
end


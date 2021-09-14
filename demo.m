% compile with
% mexOpenCV mex_SCAC.cpp mex_helper_SCAC.cpp





clear all;
close all;

currentFolder = pwd;
addpath(genpath(currentFolder));
scalespn=0.01;
Tspn=0;
if Tspn==0
    spn=600;
end



S2=1;%0=no;1=yes  when S2 is set as 1, the results are of the whole procedure. Otherwise, the results are of step one only.
ItrSet=10; %Itr
lambda=5; %lambda*100 e.g. when is set as 5, the actual lambda is 0.05.


 Files = dir(strcat('img\\','*.jpg'));

time=0.00000;
LengthFiles = length(Files);
for i = 1:LengthFiles;
    img = imread(strcat('img\',Files(i).name));
    str=['img\',Files(i).name];
    
    Row = size(img, 1); Column = size(img, 2);
    
    
    img=uint8(img);
    
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    R=uint8(R);
    G=uint8(G);
    B=uint8(B);
    
    if Tspn==1
        spn=Row*Column*scalespn;
    end
    st = clock;
    [label, spnumber] = mex_SCAC(img, spn, S2, ItrSet, lambda);
    
    deltatime=etime(clock,st);
    time=time+deltatime;
    fprintf(' spnumber=%d\n',spnumber);
   
    labelFinal=label;
   
    
    resultfile= strrep(str,'jpg', 'csv');
    csvwrite(resultfile,labelFinal);
    
    
    seeds=labelFinal;
    
    seedstemp=zeros(Row, Column);
    seedsflag=zeros(Row, Column);
    for j=2:Row-1
        for k=2:Column-1
            if seeds(j,k)~=seeds(j-1,k)
                if seedsflag(j-1,k)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
            if seeds(j,k)~=seeds(j+1,k)
                if seedsflag(j+1,k)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
            if seeds(j,k)~=seeds(j,k-1)
                if seedsflag(j,k-1)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
            if seeds(j,k)~=seeds(j,k+1)
                if seedsflag(j,k+1)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
        end
    end
    seeds=seedstemp;
    img = im2double(img);
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    R(seeds>0)=255;
    G(seeds>0)=0;
    B(seeds>0)=0;
    img(:,:,1) = R;
    img(:,:,2) = G;
    img(:,:,3) = B;
    resultImage = [str '_SCAC.png'];
    imwrite(img,resultImage);
    
    
    
    
    
    
end
avertime=time/LengthFiles;
fprintf(' Average took %.5f second\n',avertime);


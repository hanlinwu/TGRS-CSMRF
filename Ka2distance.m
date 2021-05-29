function out=Ka2distance(His1,His2) 
%calculate the \chi^2 distance
c1=(His1-His2).^2;
c2=His1+His2;
nz=find(c2~=0);
out=sum(c1(nz)./c2(nz));
end
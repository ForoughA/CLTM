CP = trainResultsc.CP([1:24,26:end]);
CPdnr = DNRtrainResults.CP([1:24,26:end]);
EP = trainResultsc.EAcp([1:24,26:end]);
EPdnr = DNRtrainResults.EAcp([1:24,26:end]);

difV = CP - CPdnr;
difE = EP - EPdnr;
impV = difV ./ CPdnr;
impE = difE ./ EPdnr;

ABv = mean(impV);
ABe = mean(impE);

MBv = median(impV);
MBe = median(impE);

BoAv = (mean(CP) - mean(CPdnr))/mean(CPdnr);
BoAe = (mean(EP) - mean(EPdnr))/mean(EPdnr);

BoMv = (median(CP) - median(CPdnr))/median(CPdnr);
BoMe = (median(EP) - median(EPdnr))/median(EPdnr);

[ABv BoAv MBv BoMv;
    ABe BoAe MBe BoMe]


%%

CA = trainResultsc.CA([1:24,26:end]);
CAdnr = DNRtrainResults.CA([1:24,26:end]);
EA = trainResultsc.EAca([1:24,26:end]);
EAdnr = DNRtrainResults.EAca([1:24,26:end]);

difV = CA - CAdnr;
difE = EA - EAdnr;
impV = difV ./ CAdnr;
impE = difE ./ EAdnr;

ABv = mean(impV);
ABe = mean(impE);

MBv = median(impV);
MBe = median(impE);

BoAv = (mean(CA) - mean(CAdnr))/mean(CAdnr);
BoAe = (mean(EA) - mean(EAdnr))/mean(EAdnr);

BoMv = (median(CA) - median(CAdnr))/median(CAdnr);
BoMe = (median(EA) - median(EAdnr))/median(EAdnr);

[ABv BoAv MBv BoMv;
    ABe BoAe MBe BoMe]

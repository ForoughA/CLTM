clear
addpath(genpath('/home/forough/dvp/synthetic_exp/UGM/'))
addpath('/home/forough/dvp/synthetic_exp/toolbox/')

load('splittedData2.mat')
load('SplittedAdjData2.mat')

load('learn_UGMboth/results9_data2')
load('learn_UGMboth/testInfo_results9_data2')

load('learn_UGMboth/Params_data2.mat')
load('learn_UGMboth/eParamsCond.mat')

load('trainCandidates')
clampedSettr = candidates(1:5);

load('testCandidates')
clampedSette = candidates(1:5);

%%
clampedSette = [];
clampedSettr = [];
% clampedSet = CorrCandidates(1:10);
% clampedSette = find(testCovariates(:,2,1)==1);
% clampedSettr = find(trainCovariates(:,2,1)==1);
% emp1 = randperm(length(clampedSettr));
% clampedSettr(sort(emp1(1:10))) = [];
% emp = randperm(length(clampedSette));
% clampedSette(sort(emp(1:10))) = [];
% clampedSet = [
%      1
%      4
%      7
%     12
%     13
%     15
%     16
%     17
%     18
%     28
%     29
%     30
%     37
%     39
%     41
%     43
%     44
%     45
%     48
%     50
%     58
%     59
%     60
%     63
%     66
%     67
%     69
%     70
%     74
%     95];

nSamples = size(testSamples,2);
clampedCLRFtr = zeros(length(adjmatT),nSamples);
clampedCLRFtr(clampedSettr,:) = trainSamples(clampedSettr,:)+1;

clampedCLRFte = zeros(length(adjmatTtest),nSamples);
clampedCLRFte(clampedSette,:) = testSamples(clampedSette,:)+1;
% % clampedCRF = zeros(length(adjmatCL{1}),nSamples);
% % clampedCRF(clampedSet,:) = Samples(clampedSet,:)+1;
clampedDNRtr = -1*ones(size(trainSamples));
clampedDNRtr(clampedSettr,:) = trainSamples(clampedSettr,:);

clampedDNRte = -1*ones(size(testSamples));
clampedDNRte(clampedSette,:) = testSamples(clampedSette,:);

[VPctr,CPctr,CActr,EActr,EAcpctr,EAcactr] = condPredAllEdge(ParamsEM,dParams,adjmatT,out_covariates,out_depCovariates,trainSamples+1,clampedCLRFtr,traineSamples,traineCovariates,eParams);
% [VPcrf,CPcrf,CAcrf,EAcrf,EAcpcrf,EAcacrf] = condPredAllEdge(ParamsCRF,dParamsCRF,adjmatCL{1},covariates,depCovariates,Samples+1,clampedCRF,eSamples,eCovariates,eParams);
[VPtr,CPtr,CAtr,EAtr,EAcptr,EAcatr] = condPredBerAllEdge(Params,trainCovariates,trainSamples,clampedDNRtr,traineSamples,traineCovariates,eParams);


[VPcte,CPcte,CActe,EActe,EAcpcte,EAcacte] = condPredAllEdge(ParamsEM,dParams,adjmatTtest,out_covariatestest,out_depCovariatestest,testSamples+1,clampedCLRFte,testeSamples,testeCovariates,eParams);
% [VPcrf,CPcrf,CAcrf,EAcrf,EAcpcrf,EAcacrf] = condPredAllEdge(ParamsCRF,dParamsCRF,adjmatCL{1},covariates,depCovariates,Samples+1,clampedCRF,eSamples,eCovariates,eParams);
[VPte,CPte,CAte,EAte,EAcpte,EAcate] = condPredBerAllEdge(Params,testCovariates,testSamples,clampedDNRte,testeSamples,testeCovariates,eParams);

% trueCP = sum(Samples,1)/nObs;

VPctr(25)=NaN;
CPctr(25)=NaN;
CActr(25)=NaN;
EActr(25)=NaN;
EAcpctr(25)=NaN;
EAcactr(25)=NaN;

VPtr(25)=NaN;
CPtr(25)=NaN;
CAtr(25)=NaN;
EAtr(25)=NaN;
EAcptr(25)=NaN;
EAcatr(25)=NaN;

VPcte(25)=NaN;
CPcte(25)=NaN;
CActe(25)=NaN;
EActe(25)=NaN;
EAcpcte(25)=NaN;
EAcacte(25)=NaN;

VPte(25)=NaN;
CPte(25)=NaN;
CAte(25)=NaN;
EAte(25)=NaN;
EAcpte(25)=NaN;
EAcate(25)=NaN;

% figure;
% subplot(1,3,1);plot(CPc,'-.*r');hold on;plot(CP,'--bo');plot(CPcrf,':ks');title('Vertex conditional presence')
% xlabel('time points');ylabel('accuracy');legend('CLRF','DNR','CRF');
% subplot(1,3,2);plot(CAc,'r');hold on;plot(CA,'b');plot(CAcrf,'k');title('Vertex conditional absence')
% xlabel('time points');ylabel('accuracy');legend('CLRF','DNR','CRF');
% subplot(1,3,3);plot(VPc,'r');hold on;plot(VP,'b');plot(VPcrf,'k');title('Vertex prediction')
% xlabel('time points');ylabel('accuracy');legend('CLRF','DNR','CRF');

figure;
subplot(1,3,1);plot(CPctr','r','LineWidth',2);hold on;plot(CPtr','k','LineWidth',2);title('train Conditional presence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,2);plot(CActr','r','LineWidth',2);hold on;plot(CAtr','k','LineWidth',2);title('train Conditional absence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,3 );plot(VPctr','r','LineWidth',2);hold on;plot(VPtr','k','LineWidth',2);title('train Vertex prediction accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')

figure;
subplot(1,3,1);plot(CPcte','r','LineWidth',2);hold on;plot(CPte','k','LineWidth',2);title('test Conditional presence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,2);plot(CActe','r','LineWidth',2);hold on;plot(CAte','k','LineWidth',2);title('test Conditional absence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,3 );plot(VPcte','r','LineWidth',2);hold on;plot(VPte','k','LineWidth',2);title('test Vertex prediction accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')


figure;
subplot(1,3,1);plot(EAcpctr','r','LineWidth',2);hold on;plot(EAcptr','k','LineWidth',2);title('train Conditional presence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,2);plot(EAcactr','r','LineWidth',2);hold on;plot(EAcatr','k','LineWidth',2);title('train Conditional absence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,3 );plot(EActr','r','LineWidth',2);hold on;plot(EAtr','k','LineWidth',2);title('train Edge prediction accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')

figure;
subplot(1,3,1);plot(EAcpcte','r','LineWidth',2);hold on;plot(EAcpte','k','LineWidth',2);title('test Conditional presence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,2);plot(EAcacte','r','LineWidth',2);hold on;plot(EAcate','k','LineWidth',2);title('test Conditional absence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')
subplot(1,3,3 );plot(EActe','r','LineWidth',2);hold on;plot(EAte','k','LineWidth',2);title('test Edge prediction accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR')

%%
[VPctr,CPctr,CActr,OCctr,co_occtr] = CLRFco_occur(ParamsEM,dParams,adjmatT,out_covariates,out_depCovariates,trainSamples+1);
[VPtr,CPtr,CAtr,OCtr,co_octr] = DNRco_occur(Params,trainCovariates,trainSamples);

nSamples = size(trainSamples,2);
nTrain = size(trainSamples,1);
% ctr = 0;
mean_CoOcctr = zeros(1,nSamples);
mean_CoOctr = zeros(1,nSamples);
for t=1:nSamples
    ctr=0;
    for i=1:nTrain
        for j=1:i-1
            mean_CoOcctr(t) = mean_CoOcctr(t) + co_occtr(i,j,t);
            mean_CoOctr(t) = mean_CoOctr(t) + co_octr(i,j,t);
        end
    end
end
mean_CoOcctr = mean_CoOcctr/(nTrain*(nTrain-1)/2);
mean_CoOctr = mean_CoOctr/(nTrain*(nTrain-1)/2);
figure;
plot(mean_CoOcctr,'r');hold on;plot(mean_CoOctr,'b');title('co-occurrence on train set')
legend('CLRF','DNR')

[VPcte,CPcte,CActe,OCcte,co_occte] = CLRFco_occur(ParamsEM,dParams,adjmatTtest,out_covariatestest,out_depCovariatestest,testSamples+1);
[VPte,CPte,CAte,OCte,co_octe] = DNRco_occur(Params,testCovariates,testSamples);

nSamples = size(testSamples,2);
nTest = size(testSamples,1);
% ctr = 0;
mean_CoOccte = zeros(1,nSamples);
mean_CoOcte = zeros(1,nSamples);
for t=1:nSamples
    ctr=0;
    for i=1:nTest
        for j=1:i-1
            mean_CoOccte(t) = mean_CoOccte(t) + co_occte(i,j,t);
            mean_CoOcte(t) = mean_CoOcte(t) + co_octe(i,j,t);
        end
    end
end
mean_CoOccte = mean_CoOccte/(nTest*(nTest-1)/2);
mean_CoOcte = mean_CoOcte/(nTest*(nTest-1)/2);
figure;
plot(mean_CoOccte,'r');hold on;plot(mean_CoOcte,'b');title('co-occurrence on test set')
legend('CLRF','DNR')


%%
figure;
subplot(1,2,1);boxplot(CPc');hold on;plot(trueCP,'--k*');title('Conditional presence accuracy CLRF')
subplot(1,2,2);boxplot(CP');hold on;plot(trueCP,'--k*');title('Conditional presence accuracyDNR')

xlabel('time points');ylabel('accuracy')
subplot(1,3,2);boxplot(CAc');hold on;boxplot(CA');title('Conditional absence accuracy')
xlabel('time points');ylabel('accuracy')
subplot(1,3,3 );boxplot(VPc');hold on;boxplot(VP');title('Vertex prediction accuracy')
xlabel('time points');ylabel('accuracy')

figure;
subplot(1,3,1);boxplot(EAcpc');subplot(1,2,2);boxplot(EAcp');title('Conditional presence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
subplot(1,3,2);boxplot(EAcac');hold on;boxplot(EAca');title('Conditional absence accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
subplot(1,3,3 );boxplot(EAc');hold on;boxplot(EA');title('Edge prediction accuracy')
xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');


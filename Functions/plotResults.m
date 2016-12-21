clear
% load('DNRvertexParamsFullTwitter/changedEdgeIter1_32.mat')
% load('DNRvertexParamsFullBeach/vertexOutCond1.mat')
% load('DNReducational/vertexOutCond4.mat')
load('DNReducational/SmallVertexOutCond5.mat')
DNRtestResults = testResults;
DNRtrainResults = trainResults;
clear DNRout testResults trainResults

nSamples = length(DNRtrainResults.CA);

for i=5
%     load(sprintf('CLRFvertexParamsFullBeach/ResultsHcov/vertexOutCond%d.mat',i))
%     load(sprintf('CLRFvertexParamsFullBeach/Results7/changedEdgeIter1_%d.mat',i))
%     load(sprintf('CLRFvertexParamsFullTwitter/Results6/changedEdgeIter2_%d.mat',i))
%     load(sprintf('CLRFvertexParamsFullTwitter/Results4/changedEdgeIter11_%d.mat',i))
%     load(sprintf('CLRFvertexParamsFullTwitter/ResultsHcov/vertexOutCond%d.mat',i))
%     load('CLRFeducational/Results4_majid/vertexOutCond10.mat')
    
%     load('CLRFeducational/Results3/vertexOutCond1.mat')
%     load('CLRFeducational/ResultsTest1_hanie/testResults.mat')
%     load('CLRFeducational/ResultsWithHhistory_Furong/results10.mat')
    load('CLRFeducational/ResultsHistory/Results1.mat')
    testResultsc = testResults;
    trainResultsc = trainResults;
    clear EMout testResults trainResults augTestData augTrainData
    
    
%     trainResultsc.CP(25) = NaN;
%     trainResultsc.CA(25) = NaN;
%     DNRtrainResults.CP(25) = NaN;
%     DNRtrainResults.CA(25) = NaN;
%     
%     trainResultsc.EAcp(25) = NaN;
%     trainResultsc.EAca(25) = NaN;
%     DNRtrainResults.EAcp(25) = NaN;
%     DNRtrainResults.EAca(25) = NaN;
    
    figure;
    plot(trainResultsc.CP,'o-r','LineWidth',2);hold on;plot(DNRtrainResults.CP,'*-k','LineWidth',2);
%     title('Vertex conditional presence')
%     xlabel('Time points');ylabel('Accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
    title('Vertex conditional presence')
%     axis([0,nSamples,0,1]);
    xlabel('Time points');ylabel('Accuracy');legend('CLRF CP','DNR CP')
    figure
%     hold on;
    plot(trainResultsc.CA,'s-r','LineWidth',2);hold on;plot(DNRtrainResults.CA,'d-k','LineWidth',2);
    title('Vertex conditional absence')
%     axis([0,nSamples,0,1]);
    xlabel('Time points');ylabel('Accuracy');legend('CLRF CA','DNR CA');
%     subplot(1,3,3);plot(trainResultsc.VP,'r','LineWidth',2);hold on;plot(DNRtrainResults.VP,'k','LineWidth',2);title('train Vertex prediction')
%     xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
    
%     CPcTr(i) = mean(trainResultsc.CP);
%     CPdnrTr(i) = mean(DNRtrainResults.CP);
%     
%     CAcTr(i) = mean(trainResultsc.CA);
%     CAdnrTr(i) = mean(DNRtrainResults.CA);
%     
%     VPcTr(i) = mean(trainResultsc.VP);
%     VPdnrTr(i) = mean(DNRtrainResults.VP);

%     figure;
%     plot(trainResultsc.EAcp,'o-r','LineWidth',2);hold on;plot(DNRtrainResults.EAcp,'*-k','LineWidth',2);
    
%     title('Edge conditional presence')
%     xlabel('time points');ylabel('Accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
%     subplot(1,2,2);

%     plot(trainResultsc.EAca,'s-r','LineWidth',2);hold on;plot(DNRtrainResults.EAca,'d-k','LineWidth',2);
%     title('Edge conditional presence and absence')
%     xlabel('Time points');ylabel('Accuracy');legend('CLRF EP','DNR EP','CLRF EA','DNR EA');
%     axis([0,nSamples,0,1]);
    
%     subplot(1,3,3);plot(trainResultsc.EA,'r','LineWidth',2);hold on;plot(DNRtrainResults.EA,'k','LineWidth',2);title('train Edge prediction')
%     xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
    
%     EAcpcTr(i) = mean(trainResultsc.EAcp);
%     EAcpdnrTr(i) = mean(DNRtrainResults.EAcp);
%     
%     EAcacTr(i) = mean(trainResultsc.EAca);
%     EAcadnrTr(i) = mean(DNRtrainResults.EAca);
%     
%     EAcTr(i) = mean(trainResultsc.EA);
%     EAdnrTr(i) = mean(DNRtrainResults.EA);
    
    figure;
    plot(testResultsc.CP,'o-r','LineWidth',2);hold on;plot(DNRtestResults.CP,'*-k','LineWidth',2);title('test Vertex conditional presence')
    xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,20,0,1]);
    figure;
    plot(testResultsc.CA,'s-r','LineWidth',2);hold on;plot(DNRtestResults.CA,'d-k','LineWidth',2);title('test Vertex conditional absence')
    xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,20,0,1]);
%     subplot(1,3,3);plot(testResultsc.VP,'r','LineWidth',2);hold on;plot(DNRtestResults.VP,'k','LineWidth',2);title('test Vertex prediction')
%     xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
    
%     CPcTe(i) = mean(testResultsc.CP);
%     CPdnrTe(i) = mean(DNRtestResults.CP);
%     
%     CAcTe(i) = mean(testResultsc.CA);
%     CAdnrTe(i) = mean(DNRtestResults.CA);
%     
%     VPcTe(i) = mean(testResultsc.VP);
%     VPdnrTe(i) = mean(DNRtestResults.VP);


%     figure;
%     subplot(1,3,1);plot(testResultsc.EAcp,'r','LineWidth',2);hold on;plot(DNRtestResults.EAcp,'k','LineWidth',2);title('test Edge conditional presence')
%     xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
%     subplot(1,3,2);plot(testResultsc.EAca,'r','LineWidth',2);hold on;plot(DNRtestResults.EAca,'k','LineWidth',2);title('test Edge conditional absence')
%     xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
%     subplot(1,3,3);plot(testResultsc.EA,'r','LineWidth',2);hold on;plot(DNRtestResults.EA,'k','LineWidth',2);title('test Edge prediction')
%     xlabel('time points');ylabel('accuracy');legend('CLRF','DNR');
%     axis([0,nSamples,0,1]);
    
%     EAcpcTe(i) = mean(testResultsc.EAcp);
%     EAcpdnrTe(i) = mean(DNRtestResults.EAcp);
%     
%     EAcacTe(i) = mean(testResultsc.EAca);
%     EAcadnrTe(i) = mean(DNRtestResults.EAca);
%     
%     EAcTe(i) = mean(testResultsc.EA);
%     EAdnrTe(i) = mean(DNRtestResults.EA);
    
%     figure;
%     subplot(1,2,1);plot(trainResultsc.mean_CoOc,'r','LineWidth',2);hold on;plot(DNRtrainResults.mean_CoOc,'k','LineWidth',2);
%     xlabel('Time points');ylabel('Co-occurrence');title('Train co-occurrence');legend('CLRF','DNR')
    
%     subplot(1,2,2);plot(testResultsc.mean_CoOc,'r','LineWidth',2);hold on;plot(DNRtestResults.mean_CoOc,'k','LineWidth',2);
%     xlabel('time points');ylabel('co-occurrence');title('test co-occurrence');legend('CLRF','DNR')
    
end

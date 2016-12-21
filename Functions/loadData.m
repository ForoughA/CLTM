function [Data,covType,depCovType] = loadData(dataset)
    switch dataset
        case 'Beach'
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/splittedData2.mat')
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/SplittedAdjData2.mat')
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/covType.mat')
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/depCovType.mat')
        case 'FullBeach'
            load('/Users/Forough/Desktop/dvp/Beach/Data/Samples.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/covariates.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/eSamples.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/eCovariates.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/depCovariates.mat')
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/covType.mat')
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/depCovType.mat')
            trainSamples = Samples;
            testSamples = Samples;
            trainCovariates = covariates;
            testCovariates = covariates;
            trainDepCovariates = depCovariates;
            testDepCovariates = depCovariates;
            traineSamples = eSamples;
            testeSamples = eSamples;
            traineCovariates = eCovariates;
            testeCovariates = eCovariates;
        case 'LalehBeach'
            load('/Users/Forough/Desktop/dvp/Beach/Data/Samples.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/covariates.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/eSamples.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/eCovariates.mat')
            load('/Users/Forough/Desktop/dvp/Beach/Data/depCovariates.mat')
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/covType.mat')
            load('/Users/Forough/Desktop/dvp/Beach/DataSplit/depCovType.mat')
            trainSamples = Samples(1:5,:);
            testSamples = Samples(1:5,:);
            trainCovariates = covariates(1:5,:,:);
            testCovariates = covariates(1:5,:,:);
            trainDepCovariates = depCovariates(1:5,1:5,:,:);
            testDepCovariates = depCovariates(1:5,1:5,:,:);
            traineSamples = eSamples;
            testeSamples = eSamples;
            traineCovariates = eCovariates;
            testeCovariates = eCovariates;
        case 'Twitter'
            load('/Users/Forough/Desktop/dvp/Twitter/DataSplit/splittedData.mat')
%             load('/home/forough/dvp/Twitter/DataSplit/SplittedAdjData.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataSplit/covType.mat')%to be added
            load('/Users/Forough/Desktop/dvp/Twitter/DataSplit/depCovType.mat')
        case 'FullTwitter'
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/vertices_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/covariates_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/depCovariates_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/adj_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/eCovariates_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/covType.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/depCovType.mat')
            trainSamples = vertices;
            testSamples = vertices;
            trainCovariates = covariates;
            testCovariates = covariates;
            trainDepCovariates = depCovariates;
            testDepCovariates = depCovariates;
            traineSamples = adj;
            testeSamples = adj;
            traineCovariates = eCovariates;
            testeCovariates = eCovariates;
        case 'FullTwitter1'
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/vertices_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/covariates_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/depCovariates_week.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/adj_week.mat')
            load('/Users/Forough/Desktop/dvp/CleanedUpCodes/CLRFvertexParamsFullTwitter/Results6/ehCovariates1.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/covType.mat')
            load('/Users/Forough/Desktop/dvp/Twitter/DataWeek/depCovType.mat')
            trainSamples = vertices;
            testSamples = vertices;
            trainCovariates = covariates;
            testCovariates = covariates;
            trainDepCovariates = depCovariates;
            testDepCovariates = depCovariates;
            traineSamples = adj;
            testeSamples = adj;
            traineCovariates = ehCovariates;
            testeCovariates = ehCovariates;
            
        case 'Educational'
            load('/Users/Forough/Desktop/dvp/EducationalData/Data/completeData.mat')
            load('/Users/Forough/Desktop/dvp/EducationalData/Data/binaryCov.mat')
            load('/Users/Forough/Desktop/dvp/EducationalData/Data/smallCov.mat')
            load('/Users/Forough/Desktop/dvp/EducationalData/Data/covType.mat')
            load('/Users/Forough/Desktop/dvp/EducationalData/Data/smallCovType.mat')
            load('/Users/Forough/Desktop/dvp/EducationalData/Data/KCuniqueCat.mat')
            % converting count data to Gaussian using:
            % sqrt(# observed) (ignore the rest of the comment)
            % for now I will set c=0, but we might need to set it to
            % another number like 1/4 (These are called variance
            % stabilizing transformations.) We might even need to apply a
            % function later on to \sqrt (like arcsin for example)
            Data.trainSamples = zeros(1595, 92);
            tmp = zeros(1595,92);
            for i=1:1595
                for t=1:92
                    if sum(dataSeq(i+1,:,t)) ~= 0
                        Data.trainSamples(i,t) = sqrt(dataSeq(i+1,1,t)/...
                            sum(dataSeq(i+1,:,t)));
                    end
                    
                    if dataSeq(i+1,2,t) ~= 0
                        tmp(i,t) = sqrt(dataSeq(i+1,1,t)/...
                            dataSeq(i+1,2,t));
                    else
                        tmp(i,t) = sqrt(dataSeq(i+1,1,t));
                    end
                end
            end
%             Data.trainSamples(:,2,:) = [];
%             Data.trainSamples = reshape(Data.trainSamples,[1595,92]);
            Data.testSamples = Data.trainSamples;
            Data.trainCovariates = binaryCov;
            Data.testCovariates = binaryCov;
            Data.tmp = tmp;
            depCovType = [];%for now
            Data.smallCov = smallCov;
            Data.smallCovType = smallCovType;
            Data.KClabels = KCuniqueCat;
            
            figure;
            subplot(2,2,1);imagesc(reshape(sqrt(dataSeq(2:end,1,:)),[1595,92]))
            title('sqrt of correct count data')
            subplot(2,2,2);imagesc(reshape(sqrt(dataSeq(2:end,2,:)),[1595,92]))
            title('sqrt of incorrect count data')
            subplot(2,2,3);imagesc(Data.trainSamples)
            title('sqrt of ratio of correct')
            subplot(2,2,4);imagesc(tmp)
%             title('sqrt of ratio of correct to incorrect')
    case 'EducationalSmall'
        load('/Users/Forough/Desktop/dvp/EducationalData/CMU/OLI_Psychology/byTransactionFull/Data266/Data.mat')
        load('/Users/Forough/Desktop/dvp/EducationalData/CMU/OLI_Psychology/byTransactionFull/Data266/KCpsychCat.mat')
        Data.trainSamples = zeros(226, 92);
            tmp = zeros(226,92);
            for i=1:226
                for t=1:92
                    if sum(dataSeq(i+1,:,t)) ~= 0
                        Data.trainSamples(i,t) = sqrt(dataSeq(i+1,1,t)/...
                            sum(dataSeq(i+1,:,t)));
                    end
                    
                    if dataSeq(i+1,2,t) ~= 0
                        tmp(i,t) = sqrt(dataSeq(i+1,1,t)/...
                            dataSeq(i+1,2,t));
                    else
                        tmp(i,t) = sqrt(dataSeq(i+1,1,t));
                    end
                end
            end
            
%             figure;
%             subplot(2,2,1);imagesc(reshape(sqrt(dataSeq(2:end,1,:)),[226,92]))
%             title('sqrt of correct count data')
%             subplot(2,2,2);imagesc(reshape(sqrt(dataSeq(2:end,2,:)),[226,92]))
%             title('sqrt of incorrect count data')
%             subplot(2,2,3);imagesc(Data.trainSamples)
%             title('sqrt of ratio of correct')
%             subplot(2,2,4);imagesc(tmp)
%             title('sqrt of ratio of correct to incorrect')
            
            Data.testSamples = Data.trainSamples;
            Data.trainCovariates = binaryCov;
            Data.testCovariates = binaryCov;
            Data.tmp = tmp;
            Data.nodeLabels = KCpsychCat(2:end);
            depCovType = [];%for now
            
        case 'Student'
            load('data/samplesAll.mat')
            Data.trainSamples = samples + 1;
            Data.testSamples = samples + 1;
            Data.trainCovariates = binaryCov;
            Data.testCovariates = binaryCov;
            Data.trainCovariatesEM = covariates;
            Data.trainDepCovariates = depCovariates;
            Data.testDepCovariates = depCovariates;
            Data.bCovType = bCovType;
            
            case 'StudentSmall'
            load('data/samplesAllSmall.mat')
            Data.trainSamples = samples + 1;
            Data.testSamples = samples + 1;
            Data.trainCovariates = binaryCov;
            Data.testCovariates = binaryCov;
            Data.trainCovariatesEM = covariates;
            Data.trainDepCovariates = depCovariates;
            Data.testDepCovariates = depCovariates;
            Data.bCovType = bCovType;
            
            case 'StudentSmallCovBig'
            load('data/samplesSmallBigCov.mat')
            Data.trainSamples = samples + 1;
            Data.testSamples = samples + 1;
            Data.trainCovariates = covariates;
            Data.testCovariates = covariates;
            Data.trainCovariatesbinary = binaryCov;
%             Data.trainDepCovariates = depCovariates;
%             Data.testDepCovariates = depCovariates;
            Data.bCovType = bCovType;
            
            case '3stateSmall'
            load('data/3stateDataNew.mat')
            Data.trainSamples = samples + 1;
            Data.testSamples = samples + 1;
            Data.trainCovariates = binaryCov;
            Data.testCovariates = binaryCov;
            Data.trainCovariatesEM = covariates;
            Data.trainDepCovariates = depCovariates;
            Data.testDepCovariates = depCovariates;
            Data.bCovType = bCovType;
    end
    
    if ~ (strcmp(dataset, 'Educational') || strcmp(dataset, 'EducationalSmall') ...
            || strcmp(dataset, 'Student') || strcmp(dataset, 'StudentSmall')...
            || strcmp(dataset,'StudentSmallCovBig') || strcmp(dataset,'3stateSmall'))
        Data.trainSamples = trainSamples + 1;
        Data.testSamples = testSamples + 1;
        Data.trainCovariates = trainCovariates;
        Data.testCovariates = testCovariates;
        Data.trainDepCovariates = trainDepCovariates;
        Data.testDepCovariates = testDepCovariates;
        Data.traineSamples = traineSamples;
        Data.testeSamples = testeSamples;
        Data.traineCovariates = traineCovariates;
        Data.testeCovariates = testeCovariates;        
    end
end

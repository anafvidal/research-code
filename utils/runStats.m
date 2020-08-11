function  runStats(direc)
resultList=dir(fullfile([ direc '/*_results.mat']));
             resultList={resultList.name};
            %Average MSE
            mseAvg=0;
            mseAvgThetaLast=0;
            timeAvg=0;
            clear('mseList');            
            clear('timeList');
            for iv=1:length(resultList)
                res=load([ direc '/' resultList{iv} ]);
                mseList(iv)=res.mse;
                if isfield(res, 'execTimeFindTheta')                   
                   timeList(iv)=res.execTimeFindTheta;
                end   
            end
            if exist('mseList','var')
                mseAvg=mean(mseList);                
                standardDev=std(mseList);                 
                
                %save mse
                mseName=[direc '/mseAvg_' char(num2str(mseAvg)) '.mat'];
                save(char(mseName), 'mseAvg');                
                 mseName=[direc '/mseAvg.mat'];
                save(char(mseName), 'mseAvg');
                %save std
                stdName=[ direc '/std_' char(num2str(standardDev)) '.mat'];
                save(char(stdName), 'standardDev');
                stdName=[ direc '/std.mat'];
                save(char(stdName), 'standardDev');
                %save time
                if isfield(res, 'execTimeFindTheta')
                    timeAvg=mean(timeList);
                    timeStd=std(timeList);
                    timeName=[ direc '/timeAvg_' char(num2str(timeAvg)) '.mat'];
                    save(char(timeName), 'timeAvg');
                    timeName=[direc '/timeAvg.mat'];
                    save(char(timeName), 'timeAvg');
                end
                fprintf('mse=%d\n',mseAvg)                
                fprintf('std=%d\n',standardDev)
                fprintf('avg time in min=%d\n',timeAvg/60)
            end
       
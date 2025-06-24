% SLM CALIBRATION SCRIPT
% last updated: January 2025 by v.chambouleyron (from m. Van Kooten)%% Folder to save data

folder_name = 'D:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT calib\20250116\';%% CONNECT TO CAMERA% Find the camera you want to use using

% >> imaqhwinfo('pointgrey')
% You may have to test all of them (using code right below - it changes at every startup of matlab...) to see which one is the good one (any
% FLIR PSF image works!) 

%{

camera_number = 3;
vid = videoinput('pointgrey', camera_number);%, 'F7_Mono8_1288x964_Mode0');
src = getselectedsource(vid);
average_frames=50;
vid.FramesPerTrigger = average_frames;
triggerconfig(vid,'manual');
src.Shutter = 4.0;
start(vid);
trigger(vid);
data=getdata(vid);
mean_image=mean(data(:,:,1,:),4);
figure;
imagesc(mean_image)

%}

camera_number = 3;

% parameters to crop image
nPx = 200;
x_center = 811;
y_center =  1499;

%% CONNECT TO SLM

% Load the DLL

% Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
% should all be located in the same directory as the program referencing the
% library

if ~libisloaded('Blink_C_wrapper')
    loadlibrary('D:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\Blink_C_wrapper.dll', 'D:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\Blink_C_wrapper.h');
end

% This loads the image generation functions
if ~libisloaded('ImageGen')
    loadlibrary('D:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\ImageGen.dll', 'D:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\ImageGen.h');
end

% Basic parameters for calling Create_SDK

%for the 1920 use 12, for the small 512x512 use 8
bit_depth = 12;

num_boards_found = libpointer('uint32Ptr', 0);
constructed_okay = libpointer('int32Ptr', 0);
is_nematic_type = 1;
RAM_write_enable = 1;
use_GPU = 0;
max_transients = 10;

% This feature is user-settable; use 1 for 'on' or 0 for 'off'
wait_For_Trigger = 0; 

external_Pulse = 0;
timeout_ms = 5000;

%This parameter is specific to the small 512 with Overdrive, do not edit
reg_lut = libpointer('string'); 

% Call the constructor
calllib('Blink_C_wrapper', 'Create_SDK', bit_depth, num_boards_found, constructed_okay, is_nematic_type, RAM_write_enable, use_GPU, max_transients, reg_lut);

figure(1)
% constructed okat returns 0 for success, nonzero integer is an error
if constructed_okay.value ~= 0
    disp('Blink SDK was not successfully constructed');
    disp(calllib('Blink_C_wrapper', 'Get_last_error_message'));
    calllib('Blink_C_wrapper', 'Delete_SDK');
else
    board_number = 1;
    disp('Blink SDK was successfully constructed');
    fprintf('Found %u SLM controller(s)\n', num_boards_found.value);    
    
    % To measure the raw response we want to disable the LUT by loading a linear LUT
    lut_file = 'C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\12bit_linear.LUT';
    calllib('Blink_C_wrapper', 'Load_LUT_file',board_number, lut_file);    
    
    %set some dimensions
    height = calllib('Blink_C_wrapper', 'Get_image_height', board_number);
    width = calllib('Blink_C_wrapper', 'Get_image_width', board_number);
    NumDataPoints = 256;
    NumRegions = 1;

    %allocate arrays for our images
    Image = libpointer('uint8Ptr', zeros(width*height,1));   
    
    % Create an array to hold measurements from the analog input (AI) board
    AI_Intensities = zeros(NumDataPoints,2);    
    
    % Generate a blank wavefront correction image, you should load your
    % custom wavefront correction that was shipped with your SLM.
    PixelValue = 0;
    calllib('ImageGen', 'Generate_Solid', Image, width, height, PixelValue);
    calllib('Blink_C_wrapper', 'Write_image', board_number, Image, width*height, wait_For_Trigger, external_Pulse, timeout_ms);
	calllib('Blink_C_wrapper', 'ImageWriteComplete', board_number, timeout_ms);
	
    %%% lets setup our Point grey camera
    vid = videoinput('pointgrey', camera_number);
    src = getselectedsource(vid);
    average_frames=50;
    vid.FramesPerTrigger = average_frames;
    triggerconfig(vid,'manual');
    src.Shutter = 4.0;
    
    %find the zeroth order which should be the max.
    start(vid);
    trigger(vid);
    pause(5);
    data=getdata(vid);
    mean_image=mean(data(:,:,1,:),4);
    mean_image = mean_image(x_center-nPx:x_center+nPx,y_center-nPx:y_center+nPx);
    maximum=max(max(mean_image));
    [x,y]=find(mean_image==maximum);
    imagesc(mean_image)
    disp('Maximum found at:')
    disp(x)
    disp(y)    
    PixelsPerStripe = 16;
    
    %loop through each region
    for Region = 0:(NumRegions-1)        
        AI_Index = 1;
        
        %loop through each graylevel
        for Gray = 0:(NumDataPoints-1)
            disp(Gray)
            
            %Generate the stripe pattern and mask out current region
            calllib('ImageGen', 'Generate_Stripe', Image, width, height, PixelValue, Gray, PixelsPerStripe);
            calllib('ImageGen', 'Mask_Image', Image, width, height, Region, NumRegions);            %write the image
            calllib('Blink_C_wrapper', 'Write_image', board_number, Image, width*height, wait_For_Trigger, external_Pulse, timeout_ms);            
            
            %let the SLM settle for 10 ms
            pause(0.1);            
            
            %YOU FILL IN HERE... FIRST: read from your specific AI board, note it might help to clean up noise to average several readings
            start(vid);
            trigger(vid);
            pause(3);
            data=getdata(vid);
            mean_image=mean(data(:,:,1,:),4);
            mean_image = mean_image(x_center-nPx:x_center+nPx,y_center-nPx:y_center+nPx);
            disp('Max')
            disp(mean_image(x,y))
            imagesc(mean_image);colorbar;
            
            %SECOND: store the measurement in your AI_Intensities array
            AI_Intensities(AI_Index, 1) = Gray; 
            
            %This is the varable graylevel you wrote to collect this data point
            AI_Intensities(AI_Index, 2) = mean_image(x,y); % HERE YOU NEED TO REPLACE 0 with YOUR MEASURED VALUE FROM YOUR ANALOG INPUT BOARD            
            AI_Index = AI_Index + 1;  

        end        
        % dump the AI measurements to a csv file
        filename = [strcat(folder_name,'Raw') num2str(Region) '.csv'];
        csvwrite(filename, AI_Intensities);

	    % Always call Delete_SDK before exiting
    calllib('Blink_C_wrapper', 'Delete_SDK');
    
    end
end

stop(vid);

%destruct
if libisloaded('Blink_C_wrapper')
    unloadlibrary('Blink_C_wrapper');
endif libisloaded('ImageGen')
    unloadlibrary('ImageGen');
end
delete(vid)

%clear
%close(gcf)
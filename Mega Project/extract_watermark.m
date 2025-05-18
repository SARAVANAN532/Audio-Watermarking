    [x,fs] = audioread('output audio trim.wav');
    L = length(x);
    N = 512;
    num_frame =floor(L/N);
    
    im = uint8(zeros(h,w,3));
    
    S = size(im);
    row = 1;
    column = 1;
    
    imsg = dec2bin(0,8);
    
    count2 = 0;
    mask_no_ex = zeros(1,num_frame);
    
    for mm = 1:num_frame
        tt = (mm-1)*N+1 : mm*N;
        X = fft(x(tt), N);
        M = find_mask(X,fs);
        
        pix = uint8(zeros(1,12,3));
        S2 = size(pix);
        row2 = 1;
        column2 = 1;
        color = 1;
        count = 1;
        
        for k = 1:(N/2)
            check = 0;
            if 20*log(abs(X(k))) < M(k)
                count2 = count2+1;
                if abs(X(k)) < exp(-1.8)
                    check = 1;
                end
                
                if abs(X(k)) >= exp(-2.1)
                    imsg_tmp1 = '1';
                    imsg_tmp2 = '1';
                elseif abs(X(k)) >= exp(-4.1)
                    imsg_tmp1 = '1';
                    imsg_tmp2 = '0';
                elseif abs(X(k)) >= exp(-6.1)
                    imsg_tmp1 = '0';
                    imsg_tmp2 = '1';
                else
                    imsg_tmp1 = '0';
                    imsg_tmp2 = '0';
                end
                
                if check == 1
                    imsg(count) = imsg_tmp1;
                    imsg(count+1) = imsg_tmp2;
                    count = count+2;
                    if count > 8
                        count = 1;
                        if row2 <= S2(1)
                            pix(row2,column2,color) = bin2dec(imsg);
                        end
                        color = color+1;
                        if color > 3
                            color = 1;
                            column2 = column2+1;
                            if column2 > S2(2)
                                column2 = 1;
                                row2 = row2+1;
                            end
                        end
                    end
                end
            end
        end
    
        if (column+11) > S(2) && (row+1) <= S(1)
            last = S(2)-column+1;
            im(row,column:S(2),:) = pix(1,1:last,:);
            im(row+1,1:(column+11-S(2)),:) = pix(1,last+1:12,:);
            row = row +1;
            column = column+12-S(2);
        elseif (column+11) > S(2) && row == S(1)
            im(row,column:S(2),:) = pix(1,1:S(2)-column+1,:);
        elseif row <= S(1)
            im(row,column:(column+11),:) = pix;
            column = column+12;
        end
    
        mask_no_ex(mm) = count2;
        count2 = 0;
    end
    
    imwrite(im,'output img fin.jpg');


 % Compute Normalized Correlation
 [aud3, fs1] = audioread('mid freq audio input.mp3');
[masked, fs2] = audioread('denoised_embed audio mid.wav');
 minLength = min(length(aud3), length(masked));
aud3 = aud3(1:minLength);   % Trim to the shortest length if needed
masked = masked(1:minLength);
aud3 = aud3(:);
masked = masked(:);

rho = corr(aud3, masked);

% Display the Result
fprintf('Normalized Correlation between original and masked audio: %.4f\n', rho);


% Load the original and extracted images
originalImage = imread('input img1.jpg');  % Replace with actual filename
extractedImage = imread('output img.jpg');  % Replace with actual filename

% Ensure both images are the same size
if size(originalImage) ~= size(extractedImage)
    error('Original and extracted images must be the same size for PSNR calculation.');
end

% Convert images to double for accurate calculation
originalImage = double(originalImage);
extractedImage = double(extractedImage);

% Calculate Mean Squared Error (MSE)
mse = mean((originalImage(:) - extractedImage(:)).^2);

% Calculate PSNR
if mse == 0
    % If MSE is zero, the images are identical; set PSNR to infinity
    psnrValue = Inf;
else
    maxPixelValue = 255;  % Maximum pixel value for 8-bit images
    psnrValue = 10 * log10((maxPixelValue^2) / mse);
end

% Display the result
fprintf('PSNR between the original and extracted image: %.2f dB\n', psnrValue);
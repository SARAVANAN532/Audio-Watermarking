function [height, width] = getImageDimensions(image)

    % Get image size
    imageDims = size(image);
    
    % Extract dimensions
    height = imageDims(1);
    width = imageDims(2);
    
end
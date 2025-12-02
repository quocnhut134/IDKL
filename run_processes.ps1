Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

Set-Location -Path 'D:\Workspace\Working\IDKL'
git checkout -b run1
python train.py --cfg ./configs/SYSU.yml --p_size 4 --k_size 4
git add .
git commit -m "Complete run with p_size=4 and k_size=4"
git push origin run1
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\new_settings\run1"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Remove-Item -Path "$SourcePath\*" -Recurse -Force
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
Remove-Item -Path "$SourcePath\*" -Recurse -Force

Set-Location -Path 'D:\Workspace\Working\IDKL'
git checkout -b run2
python train.py --cfg ./configs/SYSU.yml --p_size 4 --k_size 3
git add .
git commit -m "Complete run with p_size=4 and k_size=3"
git push origin run2
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\new_settings\run2"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Remove-Item -Path "$SourcePath\*" -Recurse -Force
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
Remove-Item -Path "$SourcePath\*" -Recurse -Force

# Set-Location -Path 'D:\Workspace\Working\IDKL'
# git checkout -b run3
# python train.py --cfg ./configs/SYSU.yml --p_size 4 --k_size 2
# git add .
# git commit -m "Complete run with p_size=4 and k_size=2"
# git push origin run3
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# $DestinationPath = "D:\Workspace\Working\new_settings\run3"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# $SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# Remove-Item -Path "$SourcePath\*" -Recurse -Force
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# Remove-Item -Path "$SourcePath\*" -Recurse -Force

# Set-Location -Path 'D:\Workspace\Working\IDKL'
# git checkout -b run4
# python train.py --cfg ./configs/SYSU.yml --p_size 3 --k_size 4
# git add .
# git commit -m "Complete run with p_size=3 and k_size=4"
# git push origin run4
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# $DestinationPath = "D:\Workspace\Working\new_settings\run4"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# $SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# Remove-Item -Path "$SourcePath\*" -Recurse -Force
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# Remove-Item -Path "$SourcePath\*" -Recurse -Force

# Set-Location -Path 'D:\Workspace\Working\IDKL'
# git checkout -b run5
# python train.py --cfg ./configs/SYSU.yml --p_size 2 --k_size 4
# git add .
# git commit -m "Complete run with p_size=2 and k_size=4"
# git push origin run5
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# $DestinationPath = "D:\Workspace\Working\new_settings\run5"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# $SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# Remove-Item -Path "$SourcePath\*" -Recurse -Force
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# Remove-Item -Path "$SourcePath\*" -Recurse -Force

# Set-Location -Path 'D:\Workspace\Working\IDKL'
# git checkout -b run6
# python train.py --cfg ./configs/SYSU.yml --p_size 2 --k_size 2
# git add .
# git commit -m "Complete run with p_size=2 and k_size=2"
# git push origin run6
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# $DestinationPath = "D:\Workspace\Working\new_settings\run6"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# $SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# Remove-Item -Path "$SourcePath\*" -Recurse -Force
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# Remove-Item -Path "$SourcePath\*" -Recurse -Force

# Set-Location -Path 'D:\Workspace\Working\IDKL'
# git checkout -b run7
# python train.py --cfg ./configs/SYSU.yml --p_size 2 --k_size 1
# git add .
# git commit -m "Complete run with p_size=2 and k_size=1"
# git push origin run7
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# $DestinationPath = "D:\Workspace\Working\new_settings\run7"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# $SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# Remove-Item -Path "$SourcePath\*" -Recurse -Force
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# Remove-Item -Path "$SourcePath\*" -Recurse -Force

# Set-Location -Path 'D:\Workspace\Working\IDKL'
# git checkout -b run8
# python train.py --cfg ./configs/SYSU.yml --p_size 1 --k_size 2
# git add .
# git commit -m "Complete run with p_size=1 and k_size=2"
# git push origin run8
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# $DestinationPath = "D:\Workspace\Working\new_settings\run8"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# $SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
# Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
# Remove-Item -Path "$SourcePath\*" -Recurse -Force
# $SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
# Remove-Item -Path "$SourcePath\*" -Recurse -Force


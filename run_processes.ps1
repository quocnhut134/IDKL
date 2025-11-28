Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

python train.py --cfg ./configs/SYSU.yml --p_size 2 --k_size 4
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=2 and k_size=4"
git push origin run3
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run3"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run3"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run4
python train.py --cfg ./configs/SYSU.yml --p_size 4 --k_size 3
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=4 and k_size=3"
git push origin run4
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run4"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run4"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run5
python train.py --cfg ./configs/SYSU.yml --p_size 4 --k_size 2
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=4 and k_size=2"
git push origin run5
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run5"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run5"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run6
python train.py --cfg ./configs/SYSU.yml --p_size 2 --k_size 3
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=2 and k_size=3"
git push origin run6
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run6"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run6"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run7
python train.py --cfg ./configs/SYSU.yml --p_size 3 --k_size 2
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=3 and k_size=2"
git push origin run7
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run7"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run7"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run8
python train.py --cfg ./configs/SYSU.yml --p_size 2 --k_size 2
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=2 and k_size=2"
git push origin run8
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run8"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run8"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run9
python train.py --cfg ./configs/SYSU.yml --p_size 2 --k_size 1
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=2 and k_size=1"
git push origin run9
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run9"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run9"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run10
python train.py --cfg ./configs/SYSU.yml --p_size 1 --k_size 2
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=1 and k_size=2"
git push origin run10
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run10"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run10"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'

git checkout -b run11
python train.py --cfg ./configs/SYSU.yml --p_size 1 --k_size 1
Set-Location -Path 'D:\Workspace\Working\IDKL'
git add .
git commit -m "Complete run with p_size=1 and k_size=1"
git push origin run11
$SourcePath = "D:\Workspace\Working\IDKL\checkpoints\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run11"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
$SourcePath = "D:\Workspace\Working\IDKL\logs\sysu\SYSU"
$DestinationPath = "D:\Workspace\Working\run11"
Move-Item -Path "$SourcePath\*" -Destination $DestinationPath
Set-Location -Path 'D:\Workspace\Working\IDKL'
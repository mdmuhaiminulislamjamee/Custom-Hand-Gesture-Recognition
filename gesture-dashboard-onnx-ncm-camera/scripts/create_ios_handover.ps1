[CmdletBinding()]
param(
    [ValidatePattern('^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}_ios_onnx_[0-9]{2}$')]
    [string]$Version = "v2026.09.17_ios_onnx_05"
)

$ErrorActionPreference = "Stop"

$ProjectDirectory = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$ReleaseRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectDirectory "handover_03"))
$FinalDirectory = [System.IO.Path]::GetFullPath((Join-Path $ReleaseRoot $Version))
$StagingDirectory = [System.IO.Path]::GetFullPath(
    (Join-Path $ReleaseRoot (".staging-" + [guid]::NewGuid().ToString("N")))
)
$PackageDirectory = Join-Path $StagingDirectory "ios"

function Assert-ChildPath {
    param([string]$Candidate, [string]$Parent)
    $prefix = $Parent.TrimEnd('\') + '\'
    if (-not $Candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe release path outside $Parent`: $Candidate"
    }
}

function Assert-Sequence {
    param([object[]]$Actual, [string[]]$Expected, [string]$Description)
    $actualText = [string]::Join("`n", @($Actual | ForEach-Object { [string]$_ }))
    $expectedText = [string]::Join("`n", $Expected)
    if ($actualText -cne $expectedText) {
        throw "$Description mismatch. Expected: $([string]::Join(', ', $Expected)); actual: $([string]::Join(', ', @($Actual)))"
    }
}

Assert-ChildPath -Candidate $FinalDirectory -Parent $ReleaseRoot
Assert-ChildPath -Candidate $StagingDirectory -Parent $ReleaseRoot
if (Test-Path -LiteralPath $FinalDirectory) {
    throw "Release already exists and will not be overwritten: $FinalDirectory"
}

$ExpectedClasses = @(
    "left", "right", "up", "down", "open_palm", "like", "dorsal", "ok",
    "fist", "thumb_down"
)
$ExpectedActions = [ordered]@{
    left = "Move Left"
    right = "Move Right"
    up = "Move Up"
    down = "Move Down"
    open_palm = "Enable / Disable Object Tracking"
    like = "Play / Pause"
    dorsal = "Return to Default Position"
    ok = "Start / Stop Recording"
    fist = "Mute"
    thumb_down = "Volume Down"
}

$ModelDirectory = Join-Path $ProjectDirectory "models"
$OnnxPath = Join-Path $ModelDirectory "gesture_mlp_production.onnx"
$MetadataPath = Join-Path $ModelDirectory "gesture_mlp_onnx_metadata.json"
$RuntimeConfigPath = Join-Path $ModelDirectory "gesture_mobile_runtime_config.json"
$LandmarkerPath = Join-Path $ModelDirectory "hand_landmarker.task"
$FaceGuardPath = Join-Path $ModelDirectory "blaze_face_short_range.tflite"
$SwiftPath = Join-Path $ProjectDirectory "swift\GestureFeatureExtractor.swift"
$SwiftReadmePath = Join-Path $ProjectDirectory "swift\README.md"

foreach ($requiredPath in @(
    $OnnxPath, $MetadataPath, $RuntimeConfigPath, $LandmarkerPath, $FaceGuardPath,
    $SwiftPath, $SwiftReadmePath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required iOS release input is missing: $requiredPath"
    }
}

$Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
$RuntimeConfig = Get-Content -LiteralPath $RuntimeConfigPath -Raw | ConvertFrom-Json
Assert-Sequence -Actual @($Metadata.output_class_order) -Expected $ExpectedClasses -Description "ONNX metadata class order"
Assert-Sequence -Actual @($RuntimeConfig.class_names) -Expected $ExpectedClasses -Description "Runtime-config class order"
if (-not [bool]$Metadata.parity.passed) {
    throw "The production ONNX conversion parity gate did not pass."
}
if (-not [bool]$Metadata.qualification.passed) {
    throw "The ten-command production qualification gate did not pass."
}
if (-not [bool]$RuntimeConfig.qualification.deployment_gate.passed) {
    throw "The runtime configuration does not record a passed deployment gate."
}
if (-not [bool]$RuntimeConfig.open_palm_orientation.reject_left_right_down) {
    throw "The runtime configuration must enforce upward-only Open Palm."
}
if ([double]$RuntimeConfig.open_palm_orientation.minimum_up_alignment -lt 0.70) {
    throw "The Open Palm upward-alignment gate is weaker than the iOS contract."
}

foreach ($entry in $ExpectedActions.GetEnumerator()) {
    $runtimeAction = [string]$RuntimeConfig.gesture_to_action.($entry.Key)
    if ($runtimeAction -cne $entry.Value) {
        throw "Action mapping mismatch for '$($entry.Key)'."
    }
    if ($null -ne $Metadata.gesture_to_action) {
        $metadataAction = [string]$Metadata.gesture_to_action.($entry.Key)
        if ($metadataAction -cne $entry.Value) {
            throw "Metadata action mapping mismatch for '$($entry.Key)'."
        }
    }
}

if ([string]$Metadata.probability_output -cne "probabilities") {
    throw "Metadata probability output must be named 'probabilities'."
}
if ([string]$Metadata.known_mass_output -cne "known_gesture_mass") {
    throw "Metadata open-set output must be named 'known_gesture_mass'."
}
if ([string]$Metadata.input.name -cne "landmark_features") {
    throw "Metadata input must be named 'landmark_features'."
}
if ([int]$Metadata.input.shape[-1] -ne 76) {
    throw "Metadata input width must be 76."
}

$OnnxHash = (Get-FileHash -LiteralPath $OnnxPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ([string]$Metadata.onnx_sha256 -cne $OnnxHash) {
    throw "The production ONNX hash does not match gesture_mlp_onnx_metadata.json."
}

$SwiftSource = Get-Content -LiteralPath $SwiftPath -Raw
foreach ($requiredSwiftToken in @(
    '"fist": "Mute"',
    '"thumb_down": "Volume Down"',
    'isOpenPalmPointingUpward',
    'runtimePoseAllowed'
)) {
    if (-not $SwiftSource.Contains($requiredSwiftToken)) {
        throw "Swift runtime contract is missing: $requiredSwiftToken"
    }
}

New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageDirectory "models") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageDirectory "swift") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageDirectory "reference") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageDirectory "contracts") -Force | Out-Null

try {
    Copy-Item -LiteralPath $OnnxPath -Destination (Join-Path $PackageDirectory "models\gesture_mlp_production.onnx")
    Copy-Item -LiteralPath $MetadataPath -Destination (Join-Path $PackageDirectory "models\gesture_mlp_onnx_metadata.json")
    Copy-Item -LiteralPath $RuntimeConfigPath -Destination (Join-Path $PackageDirectory "models\gesture_mobile_runtime_config.json")
    Copy-Item -LiteralPath $LandmarkerPath -Destination (Join-Path $PackageDirectory "models\hand_landmarker.task")
    Copy-Item -LiteralPath $FaceGuardPath -Destination (Join-Path $PackageDirectory "models\blaze_face_short_range.tflite")
    Copy-Item -LiteralPath $SwiftPath -Destination (Join-Path $PackageDirectory "swift\GestureFeatureExtractor.swift")
    Copy-Item -LiteralPath $SwiftReadmePath -Destination (Join-Path $PackageDirectory "README.md")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\config.py") -Destination (Join-Path $PackageDirectory "reference\config.py")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\geometry.py") -Destination (Join-Path $PackageDirectory "reference\geometry.py")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\runtime.py") -Destination (Join-Path $PackageDirectory "reference\runtime.py")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\model_runtime.py") -Destination (Join-Path $PackageDirectory "reference\model_runtime.py")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\hand_detection.py") -Destination (Join-Path $PackageDirectory "reference\hand_detection.py")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\face_guard.py") -Destination (Join-Path $PackageDirectory "reference\face_guard.py")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\follow_object.py") -Destination (Join-Path $PackageDirectory "reference\follow_object.py")
    Copy-Item -LiteralPath (Join-Path $ProjectDirectory "backend\ncm_camera.py") -Destination (Join-Path $PackageDirectory "reference\ncm_camera.py")

    $PayloadFiles = Get-ChildItem -LiteralPath $PackageDirectory -File -Recurse | ForEach-Object {
        $relative = $_.FullName.Substring($PackageDirectory.Length + 1).Replace('\', '/')
        [ordered]@{
            path = $relative
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }

    $Contract = [ordered]@{
        schema_version = 1
        release_version = $Version
        created_utc = [DateTime]::UtcNow.ToString("o")
        platform = "iOS"
        input = [ordered]@{
            name = "landmark_features"
            dtype = "float32"
            shape = @(1, 76)
        }
        outputs = [ordered]@{
            probabilities = @(1, 10)
            known_gesture_mass = @(1, 1)
        }
        output_class_order = $ExpectedClasses
        rejection_sentinel = "no_gesture"
        gesture_to_action = $ExpectedActions
        cross_platform_pose_rules = [ordered]@{
            open_palm = "wrist-to-palm axis must point upward"
            fist = "supported for either hand"
            thumb_down = "supported for either hand"
        }
        onnx_sha256 = $OnnxHash
        files = @($PayloadFiles)
    }
    $ContractPath = Join-Path $PackageDirectory "contracts\IOS_RELEASE_MANIFEST.json"
    $Contract | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ContractPath -Encoding utf8

    $ChecksumPath = Join-Path $PackageDirectory "SHA256SUMS.csv"
    # Materialize all rows before opening the output CSV. Otherwise the pipeline
    # can discover SHA256SUMS.csv after Export-Csv has opened it and try to hash
    # the file while it is still locked for writing.
    $ChecksumRows = @(Get-ChildItem -LiteralPath $PackageDirectory -File -Recurse |
        Sort-Object FullName | ForEach-Object {
        [pscustomobject]@{
            path = $_.FullName.Substring($PackageDirectory.Length + 1).Replace('\', '/')
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
        }
    })
    $ChecksumRows | Export-Csv -LiteralPath $ChecksumPath -NoTypeInformation -Encoding utf8

    $ArchivePath = Join-Path $StagingDirectory ("gesture-ios-handover-" + $Version + ".zip")
    Compress-Archive -LiteralPath $PackageDirectory -DestinationPath $ArchivePath -CompressionLevel Optimal
    Move-Item -LiteralPath $StagingDirectory -Destination $FinalDirectory

    Write-Host "Created verified ten-command iOS handover:" -ForegroundColor Green
    Write-Host $FinalDirectory
} catch {
    if (Test-Path -LiteralPath $StagingDirectory) {
        Assert-ChildPath -Candidate $StagingDirectory -Parent $ReleaseRoot
        Remove-Item -LiteralPath $StagingDirectory -Recurse -Force
    }
    throw
}

# Xbox OT-FM 2D Neural Engine (UWP C++20 / DirectX 12 / DirectML / MSIX)

Standalone native Xbox Series S & Series X Developer Mode application that trains a 2D generative vision neural network **from scratch (0% pre-trained weights)** using Optimal Transport Flow Matching (OT-FM) and a Frequency-Modulated (FM) sine-operator architecture.

## Why MSIX is Preferred Over Legacy APPX
- **Atomic Differential Updates & Integrity**: MSIX uses block-level hash verification (`AppxBlockMap.xml`) and clean containerization without registry/filesystem residue.
- **Xbox Dev Mode Compatibility**: Xbox Device Portal (`https://<XBOX_IP>:11443`) natively installs signed `.msix` packages alongside the self-signed `.cer` root certificate.
- **Modern Windows 10/11 SDK Tooling**: `AppxPackageExtension=.msix` compiles directly with Visual Studio 2022 (`v143`) + `DirectML` NuGet on `windows-latest` GitHub Actions runners.

## Zero-PC Cloud Build & Sideload Guide
1. Upload this entire folder structure (including `.github/workflows/build-xbox.yml`) to a new GitHub repository.
2. Open the **Actions** tab in GitHub -> select **Build Xbox OT-FM MSIX Package** -> click **Run workflow**.
3. Download the `Xbox-OTFM-MSIX-Release-x64` artifact ZIP from your phone or browser.
4. Open your Xbox Series S/X in **Dev Mode**, open the Xbox Device Portal URL (`https://<XBOX_IP>:11443`) on your phone, upload the `.msix` + `XboxOTFM_Cert.cer` + `DirectML.dll` under **My games & apps -> Add**.
5. Launch **Xbox OT-FM Neural Engine** on your TV and open `http://<XBOX_IP>:8080` on your phone browser to upload training images (16 MB max), download 2D dream images, and export/import brain checkpoints.

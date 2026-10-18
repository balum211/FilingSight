@echo off
title Push FilingSight to GitHub
echo ============================================================
echo  FilingSight - push project to github.com/balum211/FilingSight
echo ============================================================
where git >nul 2>nul
if errorlevel 1 (
  echo.
  echo Git is not installed. Install it from https://git-scm.com/download/win
  echo then double-click this file again.
  pause
  exit /b 1
)
echo.
echo Step 0: create an EMPTY repo named FilingSight on GitHub first
echo         (no README, no .gitignore). Opening the page now...
start https://github.com/new?name=FilingSight^&description=Week+2+RAG+over+Uber+and+Lyft+10-K+filings
pause
cd /d "%~dp0"
git config --global --add safe.directory "%CD:\=/%" >nul 2>nul
git remote remove origin >nul 2>&1
git remote add origin https://github.com/balum211/FilingSight.git
git branch -M main
echo.
echo Pushing to GitHub (a GitHub sign-in window may open - approve it) ...
git push -u origin main
if errorlevel 1 (
  echo.
  echo Push failed. Check that you are signed in to GitHub as balum211 and that the repo exists, then run this again.
  pause
  exit /b 1
)
echo.
echo Done! Code is at https://github.com/balum211/FilingSight
echo.
echo Next (optional): deploy on https://share.streamlit.io  - New app - repo balum211/FilingSight, file app.py
echo and add NEBIUS_API_KEY under Advanced settings - Secrets.
start https://github.com/balum211/FilingSight
pause

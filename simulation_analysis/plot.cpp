#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TCut.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TF1.h"
#include "TF2.h"
#include "TH1.h"
#include "TStyle.h"
#include "TChain.h"
#include "TSpectrum.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TSpectrum.h"
#include "TPaveLabel.h"
#include "TTimeStamp.h"
#include "TLegend.h"
#include "TText.h"
#include "TLine.h"
#include "TArrow.h"
#include "TFeldmanCousins.h"
#include "TLatex.h"
#include "TMultiGraph.h"
#include "TGaxis.h"
#include "TFrame.h"
#include "TPaveStats.h"
#include "TView3D.h"
#include "TPolyLine3D.h"


using namespace std;

void plots(char *filename, int eventnumber=0){
  TFile *file = TFile::Open(filename);
  TDirectory* ntuple = file->GetDirectory("ntuple");
  TTree *ABALONE = (TTree*) ntuple->Get("ABALONE");
  const int nev = ABALONE->GetEntries();
  cout << "Number of events " << nev << endl;
  
  Double_t X, Y, Z, Time, eID, pID, ptc, KE, DE;
  ABALONE->SetBranchAddress("X", &X);
  ABALONE->SetBranchAddress("Y", &Y);
  ABALONE->SetBranchAddress("Z", &Z);
  ABALONE->SetBranchAddress("Time", &Time);
  ABALONE->SetBranchAddress("Event_ID", &eID);//event number
  ABALONE->SetBranchAddress("Parent_ID", &pID);//identification of parent particle
  ABALONE->SetBranchAddress("Particle", &ptc);//identification of particle
  ABALONE->SetBranchAddress("KE", &KE);//particle kinetic energy
  ABALONE->SetBranchAddress("DE", &DE);//total deposited energy

  TH1D *hpx   = new TH1D("hpx","px distribution",100,-30,30);
  double xx[nev], yy[nev], zz[nev], tt[nev], kin[nev], dep[nev];
  int evID[nev], parID[nev], particle[nev];
  int j = 0;
  for (int i = 0; i < nev;i++) {
    ABALONE->GetEntry(i);
    hpx->Fill(X);
    evID[i] = eID;
    parID[i] = pID;
    particle[i] = ptc;
    if ( (eID == eventnumber) && (pID == 0) ){
      xx[j] = X;
      yy[j] = Y;
      zz[j] = Z;
      tt[j] = Time;
      kin[j] = KE;
      dep[j] = DE;
      j++;
    }
  }
  cout << j << endl;
    
  TGraph *xgraph = new TGraph(j, tt, xx);
  xgraph->SetTitle("");
  xgraph->GetYaxis()->SetTitle("position (mm)");
  xgraph->GetXaxis()->SetTitle("time");
  xgraph->GetYaxis()->SetRangeUser(-50,60);
  xgraph->SetMarkerColor(kBlue);
  xgraph->SetLineColor(kBlue);
  xgraph->SetMarkerSize(1);
  xgraph->SetMarkerStyle(21);
  TGraph *ygraph = new TGraph(j, tt, yy);
  ygraph->SetMarkerColor(kRed);
  ygraph->SetLineColor(kRed);
  ygraph->SetMarkerSize(1);
  ygraph->SetMarkerStyle(22);
  TGraph *zgraph = new TGraph(j, tt, zz);
  zgraph->SetMarkerColor(kGreen+2);
  zgraph->SetLineColor(kGreen+2);
  zgraph->SetMarkerSize(1);
  zgraph->SetMarkerStyle(23);
  TCanvas *c = new TCanvas("c","c");
  c->cd();
  xgraph->Draw("AP");
  ygraph->Draw("P");
  zgraph->Draw("P");
  TLegend *leg = new TLegend(0.7,0.88,0.5,0.8);
  leg->AddEntry(xgraph,"x position","P");
  leg->AddEntry(ygraph,"y position","P");
  leg->AddEntry(zgraph,"z position","P");
  leg->Draw();
  c->Update();
  c->Print("position.pdf");
  
  TGraph *kingraph = new TGraph(j, tt, kin);
  kingraph->SetTitle("");
  kingraph->GetYaxis()->SetTitle("energy (keV)");
  kingraph->GetXaxis()->SetTitle("time");
  //ngraph->GetYaxis()->SetRangeUser(-50,60);
  kingraph->SetMarkerColor(kBlue);
  kingraph->SetLineColor(kBlue);
  kingraph->SetMarkerSize(1);
  kingraph->SetMarkerStyle(21);
  TGraph *depgraph = new TGraph(j, tt, dep);
  depgraph->SetMarkerColor(kRed);
  depgraph->SetLineColor(kRed);
  depgraph->SetMarkerSize(1);
  depgraph->SetMarkerStyle(21);
  TCanvas *c1 = new TCanvas("c1","c1");
  c1->cd();
  kingraph->Draw("AP");
  depgraph->Draw("P");
  TLegend *leg1 = new TLegend(0.7,0.88,0.5,0.8);
  leg1->AddEntry(kingraph,"kinetic energy","P");
  leg1->AddEntry(depgraph,"deposited energy","P");
  leg1->Draw();
  c1->Update();
  c1->Print("energy.pdf");

  /*TPolyLine3D *line = new TPolyLine3D(j);
  for (int i = 0; i < j;i++)
    line->SetPoint(i,xx[i],yy[i],zz[i]);
  TCanvas *c0 = new TCanvas("c0","c0");
  TView3D *view = (TView3D*) TView::CreateView(1);
  view->SetRange(5,5,5,25,25,25);
  //TView *view = new TView(1);
  //view->SetRange(0,0,0,2,2,2);
  line->SetLineColor(kRed+1);
  line->SetLineWidth(2);
  line->Draw("LINE");
  c0->Update();
  c0->Print("plot3d.pdf");*/
  
}
  

using System;
using System.Linq;
using System.Text;
using System.Windows;
using System.Collections.Generic;
using VMS.TPS.Common.Model.API;
using VMS.TPS.Common.Model.Types;
using System.IO;
using System.Windows.Media.Media3D;
using System.Windows.Media;
using System.Diagnostics;

namespace VMS.TPS
{
    public class Script
    {
        public Script()
        {
        }

        public void Execute(ScriptContext context)
        {
            // Get patient id.
            Patient patient = context.Patient;

            if (patient == null)
            {
                MessageBox.Show("Please select a patient");

            }
            else
            {

                // Get plan setup id if available.
                PlanSetup planSetup = context.PlanSetup;

                string planSetupId = "";
                string planSetupUID = "";

                if (planSetup != null)
                {
                    planSetupId = context.PlanSetup.Id;
                    planSetupUID = context.PlanSetup.UID;

                }
                // Get the current user's username.
                string username = Environment.UserName;

                // Call the external executable with the additional username parameter.
                Preprocessing(context.Patient.Id, planSetupId, planSetupUID, username);
            }
        }


        public static void Preprocessing(string patientId, string planSetupId, string planSetupUID, string username)
        {

            string args = string.Format("{0} \"{1}\" \"{2}\" \"{3}\"",
                                patientId.Trim(),
                                planSetupId.Trim(),
                                planSetupUID.Trim(),
                                username.Trim());


            // Run process with argument patient id.
            Process proc = new Process();
            proc.StartInfo.UseShellExecute = true;

            // Do not create cmd window.
            proc.StartInfo.CreateNoWindow = true;

            proc.StartInfo.FileName = @"\\raoariaapps\raoariaapps$\Utilities\tb_adaption\main.exe";
            proc.StartInfo.WorkingDirectory = @"\\raoariaapps\raoariaapps$\Utilities\tb_adaption";
            proc.StartInfo.Arguments = args;

            proc.Start();
        }
    }
}
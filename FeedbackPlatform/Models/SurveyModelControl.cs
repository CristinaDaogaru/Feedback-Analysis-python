using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace FeedbackPlatform.Models
{
    public class SurveyModelControl
    {
        public Survey Survey { get; set; }
        public List<Survey> Surveys { get; set; }
        public int QuestionNumber { get; set; }

        public SurveyModelControl()
        {
            this.Surveys = new List<Survey>();
        }
    }
}
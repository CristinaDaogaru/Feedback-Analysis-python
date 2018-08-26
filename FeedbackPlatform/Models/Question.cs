using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace FeedbackPlatform.Models
{
    public class Question
    {
        [Key]
        public int Id { get; set; }
        public string Description { get; set; }
        public int SurveyId { get; set; }
        public Survey Survey { get; set; }
        
        public ICollection<Response> Responses { get; set; }
        public Response Response { get; set; }
    }
}